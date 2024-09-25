# functions for transformer tricks

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch, os, gc
import torch.nn as nn


#-------------------------------------------------------------------------------------
# functions for flashNorm, see paper https://arxiv.org/abs/2407.09577
#-------------------------------------------------------------------------------------
def merge_norm_proj(param, norm, proj):
  """merge norm weights into projection weights"""
  param[proj] = nn.Parameter(param[proj] @ torch.diag(param[norm]))  # flipped order
  # TODO: consider first converting to float64, then merge norm into projections,
  # and then convert back to float32. Example: torch.ones(4, dtype=torch.float32)


def set_norm_one(param, norm):
  """set all norm weights to 1.0"""
  len = list(param[norm].shape)[0]
  param[norm] = nn.Parameter(torch.ones(len))


def flashify_model(model):
  """merge norm weights into projection weights as per flashNorm"""
  with torch.no_grad():  # prevent autograd from tracking changes

    # copy the model's state_dict
    param = model.state_dict()

    # check if model uses fused projections as Phi-3
    fused_proj = 'model.layers.0.self_attn.qkv_proj.weight' in param

    # perform flashNorm merging for all layers
    for layer in range(model.config.num_hidden_layers):
      prefix = 'model.layers.' + str(layer) + '.'

      # merge input-layernorm into QKV projections
      norm = prefix + 'input_layernorm.weight'
      if fused_proj:
        merge_norm_proj(param, norm, prefix + 'self_attn.qkv_proj.weight')
      else:
        merge_norm_proj(param, norm, prefix + 'self_attn.q_proj.weight')
        merge_norm_proj(param, norm, prefix + 'self_attn.k_proj.weight')
        merge_norm_proj(param, norm, prefix + 'self_attn.v_proj.weight')
      set_norm_one(param, norm)

      # merge post-attention layernorm into Gate and Up projections
      norm = prefix + 'post_attention_layernorm.weight'
      if fused_proj:
        merge_norm_proj(param, norm, prefix + 'mlp.gate_up_proj.weight')
      else:
        merge_norm_proj(param, norm, prefix + 'mlp.gate_proj.weight')
        merge_norm_proj(param, norm, prefix + 'mlp.up_proj.weight')
      set_norm_one(param, norm)

    # if the model has untied embeddings, then merge 'model.norm' into 'lm_head'
    # see also https://huggingface.co/HuggingFaceTB/SmolLM-135M/discussions/15
    if model.config.tie_word_embeddings == False:
      merge_norm_proj(param, 'model.norm.weight', 'lm_head.weight')
      set_norm_one(param, 'model.norm.weight')

    # load the modified state_dict back into the model
    model.load_state_dict(param)


def flashify_repo(repo, out_dir=None):
  """convert LLM repo to flashNorm, store the new model in out_dir"""
  with torch.no_grad():  # prevent autograd from tracking changes

    # flashify model
    model = AutoModelForCausalLM.from_pretrained(repo, low_cpu_mem_usage=True)
    flashify_model(model)
    # print('DEBUG, should be all 1', model.model.layers[0].input_layernorm.weight)

    # save model in local directory 'out_dir'
    if out_dir == None:  # append '_flashNorm' if no output dir is defined
      out_dir = os.path.basename(repo) + '_flashNorm'
    model.save_pretrained(out_dir, from_pt=True)
    del model; gc.collect()  # run garbage collection

    # ditto with tokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    tok.save_pretrained(out_dir, from_pt=True)
    del tok; gc.collect()  # run garbage collection


#-------------------------------------------------------------------------------------
# functions for testing
#-------------------------------------------------------------------------------------
def hello_world(repo, max_new_tok=4):
  """run example inference of an LLM from HuggingFace repo or local directory"""
  tok = AutoTokenizer.from_pretrained(repo)
  model = AutoModelForCausalLM.from_pretrained(repo, low_cpu_mem_usage=True)
  # to use FP16 or bfloaf: torch_dtype=torch.float16, torch_dtype=torch.bfloat
  # note: FP16 is 30x slower than FP32 on my Mac M1, not sure why

  prompt = 'Once upon a time there was'
  inp = tok.encode(prompt, return_tensors='pt').to('cpu')
  out = model.generate(inp, pad_token_id=0, max_new_tokens=max_new_tok).ravel()
  print(tok.decode(out))
  del tok, model; gc.collect()  # run garbage collection
  # TODO: especially for Phi-3, set verbosity to quiet as follows
  #  transformers.logging.set_verbosity_error()


def perplexity(repo, speedup=1, bars=False):
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
  # TODO: consider using instead "with torch.no_grad():"

  tok = AutoTokenizer.from_pretrained(repo)
  model = AutoModelForCausalLM.from_pretrained(repo, low_cpu_mem_usage=True)

  # tokenize wikitext2
  test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
  encodings = tok('\n\n'.join(test['text']), return_tensors='pt')
  del tok; gc.collect()  # run garbage collection

  max_length = model.config.max_position_embeddings
  stride = max_length  # before it was 512 or max_length // 2
  seq_len = encodings.input_ids.size(1) // speedup

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
  print('ppl:', ppl)
  #print('nlls:', nlls)
  del model; gc.collect()  # run garbage collection


#-------------------------------------------------------------------------------------
# TODOs: add more functions
#-------------------------------------------------------------------------------------
# e.g. add a def to compare or diff two models / safetensors. See here:
#   - https://gist.github.com/so298/b5fc4127f161dbd65429f5756d771d88
#   - https://gist.github.com/madebyollin/034afe6670fc03966d075912cbccf797
