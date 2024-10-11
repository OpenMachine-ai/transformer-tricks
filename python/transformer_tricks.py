# tricks and tools for speeding up LLMs

import gc, os, time, torch, datasets, glob
import torch.nn as nn
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file, safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, logging, utils
try:
  from flashNorm_modeling_llama import *  # import local file if it exists
except ImportError:
  pass


#-------------------------------------------------------------------------------------
# functions for flashNorm, see paper https://arxiv.org/abs/2407.09577
#-------------------------------------------------------------------------------------
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


def merge_norm_proj(param, norm, proj, layer=0):
  """merge norm weights into projection weights"""
  n_key = weight(norm, layer)
  p_key = weight(proj, layer)
  param[p_key] = nn.Parameter(param[p_key] @ torch.diag(param[n_key])).detach()  # flipped order
  # TODO: consider first converting to float64, then merge norm into projections,
  # and then convert back to float32. Example: torch.ones(4, dtype=torch.float32)


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


def get_param(repo):
  """download all safetensor files from repo and return a single param dict"""
  snapshot_download(repo_id=repo, allow_patterns='*.safetensors', local_dir='tmp')
  param = {}
  for file in glob.glob('tmp/*.safetensors'):
    param.update(load_file(file))  # concatenate all parameters into a single dict
  return param


def save_repo(repo, param, config, dir):
  """save tokenizer, config, and param in local dir"""
  tok = AutoTokenizer.from_pretrained(repo)
  tok.save_pretrained(dir, from_pt=True)
  config.save_pretrained(dir, from_pt=True)
  save_file(param, dir + '/model.safetensors', metadata={'format': 'pt'})


def flashify_repo(repo, dir=None, bars=False, test=True):
  """convert LLM repo to flashNorm, store the new model in local dir"""
  with torch.no_grad():  # prevent autograd from tracking changes

    if dir == None:  # append '_flashNorm' if no output dir is defined
      dir = os.path.basename(repo) + '_flashNorm'

    # get config, download safetensors, and flashify params
    config = AutoConfig.from_pretrained(repo)
    param = get_param(repo)
    flashify(param, config, bars)
    if test:  # optionally, save a test-repo in directory *_test
      save_repo(repo, param, config, dir + '_test')

    # delete norm weights from param
    for layer in range(config.num_hidden_layers):
      del param[weight('Inorm', layer)]
      del param[weight('Anorm', layer)]
    if config.tie_word_embeddings == False:
      del param[weight('Hnorm')]

   # TODO:
    #config.architectures = ['LlamaForCausalLM_flashNorm']
    #config.auto_map = {'AutoModelForCausalLM': 'flashNorm_modeling_llama.LlamaForCausalLM_flashNorm'}
    #config.model_type = 'flashNorm'
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


def perplexity(repo, speedup=1, arch='AutoModelForCausalLM', bars=False, perf=False):
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
  model = eval(f'{arch}.from_pretrained(repo, low_cpu_mem_usage=True)')

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
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cpu')
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


#-------------------------------------------------------------------------------------
# misc tools
#-------------------------------------------------------------------------------------
def quiet_hf():
  """reduce verbosity of HuggingFace packages"""
  logging.set_verbosity_error()
  utils.logging.disable_progress_bar()
  datasets.disable_progress_bars()
  os.environ['TOKENIZERS_PARALLELISM'] = 'true'
  os.environ['HF_HUB_VERBOSITY'] = 'error'
  # for more env variables, see link below
  # https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables


#-------------------------------------------------------------------------------------
# TODOs: add more functions
#-------------------------------------------------------------------------------------

# print metadata of a safetensor file:
#    with safe_open('model.safetensors', framework='pt') as f:
#      for k, v in f.metadata().items():
#        print(f'{k}: {v}')

# e.g. add a def to compare or diff two models / safetensors. See here:
#   - https://gist.github.com/so298/b5fc4127f161dbd65429f5756d771d88
#   - https://gist.github.com/madebyollin/034afe6670fc03966d075912cbccf797

# misc TODOs:
#  - do we really need 'with torch.no_grad():' everywhere?
#  - do we really need garbage collection 'gc'?
#  - would 'torch.set_grad_enabled(False)' speed up things?
