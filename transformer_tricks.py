# tricks and tools for speeding up LLMs

import gc, os, time, torch, datasets, glob, shutil
import torch.nn as nn
from tqdm import tqdm
from huggingface_hub import snapshot_download, repo_exists
from safetensors.torch import load_file, save_file, safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, logging, utils
try:
  from flashNorm_modeling_llama import *  # import local file if it exists
except ImportError:
  pass


#-------------------------------------------------------------------------------------
# tools for working with safetensors and HuggingFace repos
#-------------------------------------------------------------------------------------
def quiet_hf():
  """reduce verbosity of HuggingFace"""
  logging.set_verbosity_error()
  utils.logging.disable_progress_bar()
  datasets.logging.disable_progress_bar()
  os.environ['TOKENIZERS_PARALLELISM'] = 'true'
  os.environ['HF_HUB_VERBOSITY'] = 'error'
  # for more env variables, see link below
  # https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables


def weight(name, layer=0):
  """get dictionary key of specific weight (such as Q from layer 0)"""
  layer_str = 'model.layers.' + str(layer) + '.'
  match name:
    # weights of each layer
    case 'Inorm': key = layer_str + 'input_layernorm.weight'
    case 'Anorm': key = layer_str + 'post_attention_layernorm.weight'
    case 'QKV'  : key = layer_str + 'self_attn.qkv_proj.weight'
    case 'Q'    : key = layer_str + 'self_attn.q_proj.weight'
    case 'K'    : key = layer_str + 'self_attn.k_proj.weight'
    case 'V'    : key = layer_str + 'self_attn.v_proj.weight'
    case 'O'    : key = layer_str + 'self_attn.o_proj.weight'
    case 'GU'   : key = layer_str + 'mlp.gate_up_proj.weight'
    case 'G'    : key = layer_str + 'mlp.gate_proj.weight'
    case 'U'    : key = layer_str + 'mlp.up_proj.weight'
    case 'D'    : key = layer_str + 'mlp.down_proj.weight'
    # embedding weights
    case 'Hnorm': key = 'model.norm.weight'          # normalization of lm_head
    case 'H'    : key = 'lm_head.weight'             # output embeddings
    case 'E'    : key = 'model.embed_tokens.weight'  # input embeddings
  return key


def get_param(repo, get_meta=False, tmp_dir='get_param_tmp'):
  """download all *.safetensors files from repo (or local dir) and return a single
  param dict, and optionally also return the metadata.

  Each repo downloads into its own subdirectory of tmp_dir. Sharing one
  directory silently merges models: the glob below collects every *.safetensors
  present, so a second call inherits the shards the first one left behind and
  folds them into the param dict. diff_safetensors is hit hardest, because it
  calls get_param on two repos back to back and its whole job is telling them
  apart."""

  # download and get list of files
  if repo_exists(repo):
    dir = os.path.join(tmp_dir, *repo.split('/'))  # one subdir per repo
    snapshot_download(repo_id=repo, allow_patterns='*.safetensors', local_dir=dir)
  else:  # if repo doesn't exist on HuggingFace, then 'repo' specifies local dir
    dir = repo
  files = sorted(glob.glob(dir + '/*.safetensors'))  # sorted so files[0] is stable
  if not files:
    raise FileNotFoundError(f"no *.safetensors files found for '{repo}' in '{dir}'")

  # get parameters
  param = {}
  for file in files:
    param.update(load_file(file))  # concatenate all parameters into a single dict

  # return param only, or param and metadata
  if get_meta == False:
    return param
  else:
    with safe_open(files[0], framework='pt') as f: # use the first file
      return param, f.metadata()


def save_repo(repo, param, config, dir):
  """save tokenizer, config, and param in local dir.

  Copies the source tokenizer files verbatim (instead of roundtripping through
  AutoTokenizer) to preserve format compatibility across transformers versions
  -- a roundtrip via transformers 5.x produces a tokenizer_config.json that
  transformers 4.x cannot parse, breaking stock vLLM loading."""
  os.makedirs(dir, exist_ok=True)
  if repo_exists(repo):
    # download tokenizer files to a sibling temp dir, then copy just the files
    # (snapshot_download creates a .cache/ subdir that we exclude from `dir`)
    tmp = dir + '_tok_src'
    snapshot_download(repo_id=repo,
                      allow_patterns=['tokenizer*', 'special_tokens_map*',
                                      'vocab*', 'merges*', 'chat_template*'],
                      local_dir=tmp)
    for f in os.listdir(tmp):
      src = os.path.join(tmp, f)
      if os.path.isfile(src) and not f.startswith('.'):
        shutil.copy2(src, dir)
    shutil.rmtree(tmp)
  else:
    # local source dir; fall back to AutoTokenizer roundtrip
    tok = AutoTokenizer.from_pretrained(repo)
    tok.save_pretrained(dir, from_pt=True)
  config.save_pretrained(dir, from_pt=True)
  save_file(param, dir + '/model.safetensors', metadata={'format': 'pt'})


#-------------------------------------------------------------------------------------
# functions for flashNorm, see paper https://arxiv.org/abs/2407.09577
#-------------------------------------------------------------------------------------
def merge_norm_proj(param, norm, proj, layer=0):
  """merge norm weights into projection weights.

  Computes the merge in float32 (the product of bf16 × fp32 fits exactly in
  fp32's 24-bit mantissa) and casts the result back to the projection's
  original dtype so the stored checkpoint matches config.dtype."""
  n_key = weight(norm, layer)
  p_key = weight(proj, layer)
  orig_dtype = param[p_key].dtype
  w = param[p_key].to(torch.float32)
  g = param[n_key].to(torch.float32)
  param[p_key] = nn.Parameter((w @ torch.diag(g)).to(orig_dtype)).detach()


def set_norm_one(param, norm, layer=0):
  """set all norm weights to 1.0"""
  n_key = weight(norm, layer)
  len = list(param[n_key].shape)[0]
  param[n_key] = nn.Parameter(torch.ones(len)).detach()


def flashify(param, config, bars):
  """merge norm weights into projection weights as per flashNorm"""
  with torch.no_grad():  # prevent autograd from tracking changes

    # check if model uses fused projections (such as in Phi-3)
    fused_proj = weight('QKV') in param

    # perform flashNorm merging for each layer
    for layer in tqdm(range(config.num_hidden_layers), disable=not bars):

      # merge input-layernorm into QKV projections
      if fused_proj:
        merge_norm_proj(param, 'Inorm', 'QKV', layer)
      else:
        merge_norm_proj(param, 'Inorm', 'Q', layer)
        merge_norm_proj(param, 'Inorm', 'K', layer)
        merge_norm_proj(param, 'Inorm', 'V', layer)
      set_norm_one(param, 'Inorm', layer)

      # merge post-attention layernorm 'Anorm' into Gate and Up projections
      if fused_proj:
        merge_norm_proj(param, 'Anorm', 'GU', layer)
      else:
        merge_norm_proj(param, 'Anorm', 'G', layer)
        merge_norm_proj(param, 'Anorm', 'U', layer)
      set_norm_one(param, 'Anorm', layer)

    # if the model has untied embeddings, then merge 'Hnorm' into 'lm_head'
    # see also https://huggingface.co/HuggingFaceTB/SmolLM-135M/discussions/15
    if config.tie_word_embeddings == False:
      merge_norm_proj(param, 'Hnorm', 'H')
      set_norm_one(param, 'Hnorm')


def flashify_repo(repo, dir=None, bars=False, strict=False, test=None):
  """convert LLM repo to flashNorm, store the new model in local dir.

  Default (strict=False, 'compat mode'): norm weights are folded into
  projection weights but kept in the state dict as all-ones tensors. The
  resulting checkpoint loads in stock transformers and vLLM as a regular
  model of the source architecture with no code changes.

  strict=True: norm weight tensors are deleted from the state dict, producing
  a smaller checkpoint that requires custom modeling code (e.g.
  LlamaFlashNorm) to load.

  Legacy `test=...` mode (preserved for backward compatibility with pre-
  compat-mode callers): when `test` is passed explicitly, this function
  reproduces the old behavior exactly. test=True writes a companion
  '*_test' directory with norm tensors kept as all-ones; either test=True
  or test=False also makes the main output at `dir` strict (norm tensors
  deleted), matching the pre-compat-mode main-output behavior. When `test`
  is not passed, the new `strict` argument controls behavior."""
  with torch.no_grad():  # prevent autograd from tracking changes

    if dir == None:  # append '_flashNorm' if no output dir is defined
      dir = os.path.basename(repo) + '_flashNorm'

    # get config, download safetensors, and flashify params
    config = AutoConfig.from_pretrained(repo)
    param = get_param(repo)
    flashify(param, config, bars)

    # Legacy test=True writes a companion '*_test' directory with norms kept
    # as all-ones (pre-compat-mode behavior).
    if test is True:
      save_repo(repo, param, config, dir + '_test')

    # Main output is strict when user passes strict=True or any legacy
    # `test=...` argument (the pre-compat-mode main output was always strict
    # regardless of test's value).
    if strict or test is not None:
      for layer in range(config.num_hidden_layers):
        del param[weight('Inorm', layer)]
        del param[weight('Anorm', layer)]
      if config.tie_word_embeddings == False:
        del param[weight('Hnorm')]

    save_repo(repo, param, config, dir)

    del param; gc.collect()  # run garbage collection


#-------------------------------------------------------------------------------------
# functions for testing
#-------------------------------------------------------------------------------------
def hello_world(repo, max_new_tok=4, arch='AutoModelForCausalLM', perf=False):
  """run example inference of an LLM from HuggingFace repo or local directory"""
  tok = AutoTokenizer.from_pretrained(repo)
  model = eval(f'{arch}.from_pretrained(repo, low_cpu_mem_usage=True)')
  # to use FP16 or bfloaf: torch_dtype=torch.float16, torch_dtype=torch.bfloat
  # note: FP16 is 30x slower than FP32 on my Mac M1, not sure why

  prompt = 'Once upon a time there was'
  start_time = time.perf_counter()
  inp = tok.encode(prompt, return_tensors='pt').to('cpu')
  out = model.generate(inp, pad_token_id=0, max_new_tokens=max_new_tok).ravel()
  print(tok.decode(out),
        f'  (time: {time.perf_counter() - start_time:.2f}s)' if perf else '')
  del tok, model; gc.collect()  # run garbage collection
  # TODO: especially for Phi-3, set verbosity to quiet as follows
  #  transformers.logging.set_verbosity_error()


def perplexity(repo, speedup=1, arch='AutoModelForCausalLM', bars=False, perf=False, device='cpu', dtype=None):
  """calculate perplexity of an LLM with wikitext2
  this def is copied from https://huggingface.co/docs/transformers/perplexity
  I made the following changes to adapt it for SmolLM (was GPT2 before):
    - changed model and tokenizer
    - changed 'from transformers import' to point to 'Auto*' (was 'GTP2*' before)
    - changed 'max_length' to 'config.max_position_embeddings'
    - changed 'device' from 'cuda' to 'cpu'
    - changed 'stride' to be 'max_length' (was 512 or 'max_length//2' before)
    - removed 'with torch.no_grad()' and added global 'torch.set_grad_enabled(False)'
  Perhaps a simpler and cleaner way is given here:
  https://huggingface.co/spaces/evaluate-metric/perplexity"""

  torch.set_grad_enabled(False)  # speed up torch
  # TODO: consider using instead 'with torch.no_grad():'

  tok = AutoTokenizer.from_pretrained(repo)

  kwargs = {'low_cpu_mem_usage': True}
  if dtype is not None:
    kwargs['torch_dtype'] = dtype

  model = eval(f'{arch}.from_pretrained(repo, **kwargs)').to(device)

  # tokenize wikitext2
  test = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
  encodings = tok('\n\n'.join(test['text']), return_tensors='pt')
  del tok; gc.collect()  # run garbage collection

  max_length = model.config.max_position_embeddings
  stride = max_length  # before it was 512 or max_length // 2
  seq_len = encodings.input_ids.size(1) // speedup

  start_time = time.perf_counter()
  nlls = []
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride), disable=not bars):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    outputs = model(input_ids, labels=target_ids)

    # loss is calculated using CrossEntropyLoss which averages over valid labels
    # N.B. the model only calculates loss over trg_len - 1 labels, because it
    # internally shifts the labels to the left by 1.
    neg_log_likelihood = outputs.loss
    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
      break

  ppl = torch.exp(torch.stack(nlls).mean())
  print(f'perplexity = {ppl:.3f}',
        f'  (time: {time.perf_counter() - start_time:.2f}s)' if perf else '')
  # print('nlls:', nlls)
  del model; gc.collect()  # run garbage collection
  return ppl.item()


#-------------------------------------------------------------------------------------
# debug tools
#-------------------------------------------------------------------------------------
def diff_safetensors(repo1, repo2):
  """compare differences of safetensor file(s) between repo1 and repo2"""
  param1, meta1 = get_param(repo1, get_meta=True)
  param2, meta2 = get_param(repo2, get_meta=True)
  set1, set2 = set(param1.keys()), set(param2.keys())

  # diff keys
  if set1 == set2:
    print('>>> SAFE-DIFF: both repos have the same safetensor keys')
  else:
    if set1 - set2:
      print(f'>>> SAFE-DIFF: these keys are only in repo {repo1}: {set1 - set2}')
    if set2 - set1:
      print(f'>>> SAFE-DIFF: these keys are only in repo {repo2}: {set2 - set1}')

  # diff tensors
  found_diff = False
  for key in set1.intersection(set2):
    if not torch.equal(param1[key], param2[key]):
      found_diff = True
      print(f'>>> SAFE-DIFF: tensors {key} are not equal')
  if not found_diff:
    print('>>> SAFE-DIFF: all intersecting tensors are equal')

  # diff metadata
  if meta1 == meta2:
    print('>>> SAFE-DIFF: both repos have the same safetensor metadata')
  else:
    print(f'>>> SAFE-DIFF: metadata of repo {repo1}: {meta1}')
    print(f'>>> SAFE-DIFF: metadata of repo {repo2}: {meta2}')


# misc TODOs:
#  - do we really need 'with torch.no_grad():' everywhere?
#  - do we really need garbage collection 'gc'?
#  - would 'torch.set_grad_enabled(False)' speed up things?
