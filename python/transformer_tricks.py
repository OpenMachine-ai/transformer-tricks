# functions for transformer tricks

import torch
import torch.nn as nn

#-------------------------------------------------------------------------------------
# defs for flashNorm, see paper https://arxiv.org/abs/2407.09577
#-------------------------------------------------------------------------------------
def merge_norm_proj(param, norm, proj):
  """merge norm weights into projection weights"""
  param[proj] = nn.Parameter(param[proj] @ torch.diag(param[norm]))  # flipped order
  return param
  # TODO: consider first converting to float64, then merge norm into projections,
  # and then convert back to float32. Example: torch.ones(4, dtype=torch.float32)

def set_norm_one(param, norm):
  """set all norm weights to 1.0"""
  len = list(param[norm].shape)[0]
  param[norm] = nn.Parameter(torch.ones(len))
  return param

def flashify(model):
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
        param = merge_norm_proj(param, norm, prefix + 'self_attn.qkv_proj.weight')
      else:
        param = merge_norm_proj(param, norm, prefix + 'self_attn.q_proj.weight')
        param = merge_norm_proj(param, norm, prefix + 'self_attn.k_proj.weight')
        param = merge_norm_proj(param, norm, prefix + 'self_attn.v_proj.weight')
      param = set_norm_one(param, norm)

      # merge post-attention layernorm into Gate and Up projections
      norm = prefix + 'post_attention_layernorm.weight'
      if fused_proj:
        param = merge_norm_proj(param, norm, prefix + 'mlp.gate_up_proj.weight')
      else:
        param = merge_norm_proj(param, norm, prefix + 'mlp.gate_proj.weight')
        param = merge_norm_proj(param, norm, prefix + 'mlp.up_proj.weight')
      param = set_norm_one(param, norm)

    # if the model has untied embeddings, then merge 'model.norm' into 'lm_head'
    # see also https://huggingface.co/HuggingFaceTB/SmolLM-135M/discussions/15
    if model.config.tie_word_embeddings == False:
      param = merge_norm_proj(param, 'model.norm.weight', 'lm_head.weight')
      param = set_norm_one(param, 'model.norm.weight')

    # load the modified state_dict back into the model
    model.load_state_dict(param)
  return model

#-------------------------------------------------------------------------------------
# TODOs: add more defs
#-------------------------------------------------------------------------------------
# e.g. add a def to compare or diff two models / safetensors. See here:
#   - https://gist.github.com/so298/b5fc4127f161dbd65429f5756d771d88
#   - https://gist.github.com/madebyollin/034afe6670fc03966d075912cbccf797
