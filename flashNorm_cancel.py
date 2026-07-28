# FlashNorm norm cancellation (Proposition 3 of the FlashNorm paper, arXiv:2407.09577)
#
# An RMSNorm followed by a bias-free linear layer followed by another RMSNorm makes the
# first RMSNorm redundant: after folding its gain into the linear layer (Proposition 1),
# scale invariance of the second norm cancels the first norm's 1/RMS division. This module
# applies that cancellation to loaded HuggingFace models, with an eligibility audit.
#
# Eligibility criterion: the pre-attention norm can be cancelled only if EVERY consumer of
# its output is re-normalized downstream through bias-free, positively homogeneous paths.
#   - QKV-normalized models (e.g. Gemma 4: q_norm, k_norm, v_norm per head): eligible,
#     cancellation is exact for unregularized RMS and epsilon-approximate otherwise.
#   - MLA models (DeepSeek-V2 family, MiniCPM3): NOT eligible for full cancellation, because
#     the decoupled RoPE-key slice of the kv_a projection bypasses the latent norm. Use
#     mla_partial_cancel() instead: it keeps the per-token RMS scalar but applies it only to
#     the small RoPE slice (and to the query path when there is no q_a_layernorm), which
#     still removes the norm's weight tensor and the full-width elementwise multiply.
#   - QK-only-normalized models (e.g. Gemma 3: no v_norm): NOT eligible, the value path
#     would silently change scale. cancel_pre_attention_norms() refuses unless force=True.
#
# Notes:
#   - These are runtime transforms on an instantiated model (stock modeling code always
#     executes the division, so the cancellation cannot be expressed in a checkpoint alone).
#   - Folding is computed in float64 and cast back to the parameter dtype.
#   - Norm gain conventions: 'plain' multiplies by w (ones-init, e.g. HF Gemma 4, Llama);
#     'one_plus' multiplies by 1 + w (zeros-init, e.g. HF Gemma 1-3, JAX gemma). The
#     convention is auto-detected from the weight statistics unless given explicitly.

import torch
import torch.nn as nn


def detect_gain_convention(norm):
  """Return 'one_plus' if the norm's stored weight looks zeros-initialized, else 'plain'.

  Heuristic: trained plain gains cluster near 1, trained (1+w) offsets cluster near 0.
  Pass convention='plain' or 'one_plus' explicitly to the fold functions to override."""
  w = norm.weight.detach().float()
  return 'one_plus' if w.abs().mean().item() < 0.5 else 'plain'


def _effective_gain(norm, convention):
  if convention == 'auto':
    convention = detect_gain_convention(norm)
  g = norm.weight.detach().double()
  if convention == 'one_plus':
    g = 1.0 + g
  return g, convention


def fold_norm_into_projs(norm, projs, convention='auto'):
  """Fold a norm's gain into the following bias-free linear layers (Proposition 1).

  After folding, the norm's stored weight is set to the identity for its convention
  (ones for 'plain', zeros for 'one_plus'), so the model still computes the same
  function while the gain lives inside the projection weights."""
  g, convention = _effective_gain(norm, convention)
  with torch.no_grad():
    for p in projs:
      assert isinstance(p, nn.Linear) and p.bias is None, 'fold requires bias-free nn.Linear'
      assert p.weight.shape[1] == g.shape[0], 'norm dim does not match projection input dim'
      p.weight.data = (p.weight.data.double() * g.unsqueeze(0)).to(p.weight.dtype)
    norm.weight.data = torch.zeros_like(norm.weight) if convention == 'one_plus' \
        else torch.ones_like(norm.weight)
  return convention


class _BypassedNorm(nn.Module):
  """Drop-in replacement for a cancelled RMSNorm: passes the input through unchanged.
  With keep_scalar=True it also records the per-token scalar s = 1/sqrt(mean(x^2) + eps)
  for downstream hooks (used by mla_partial_cancel)."""

  def __init__(self, eps, keep_scalar=True):
    super().__init__()
    self.eps = eps
    self.keep_scalar = keep_scalar
    self.s = None

  def forward(self, x):
    if self.keep_scalar:
      self.s = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
    return x


def _get_norm_eps(norm):
  eps = getattr(norm, 'eps', None)
  if eps is None:
    eps = getattr(norm, 'variance_epsilon', None)
  assert eps is not None, 'could not read epsilon from norm module'
  return eps


def decoder_layers(model):
  """Resolve the decoder layer list across common HF model nestings."""
  for path in ('model.layers', 'model.language_model.layers', 'language_model.model.layers'):
    obj = model
    try:
      for part in path.split('.'):
        obj = getattr(obj, part)
      return obj
    except AttributeError:
      continue
  raise AttributeError('could not locate decoder layers; pass them explicitly')


def audit_pre_attention_cancellation(model, layers=None):
  """Per-layer eligibility report for cancelling input_layernorm before attention.

  A layer is eligible if every present q/k/v projection has a matching q/k/v norm.
  KV-shared layers (projection absent or None) are eligible through the present paths."""
  layers = decoder_layers(model) if layers is None else layers
  report = []
  for i, ly in enumerate(layers):
    at = ly.self_attn
    row = {'layer': i}
    ok = True
    for name in ('q', 'k', 'v'):
      proj = getattr(at, f'{name}_proj', None)
      has_norm = getattr(at, f'{name}_norm', None) is not None
      row[f'{name}_proj'] = proj is not None
      row[f'{name}_norm'] = has_norm
      if proj is not None and not has_norm:
        ok = False
    if not any(row[f'{n}_proj'] for n in ('q', 'k', 'v')):
      ok = False  # nothing to fold into: MLA and other layouts must not pass vacuously
    row['mla'] = getattr(at, 'kv_a_proj_with_mqa', None) is not None
    row['eligible'] = ok
    report.append(row)
  return report


def cancel_pre_attention_norms(model, convention='auto', layers=None, force=False):
  """Cancel input_layernorm on every eligible decoder layer (Proposition 3).

  Folds the norm gain into the present q/k/v projections, then replaces the norm with a
  pass-through module. Raises on ineligible layers unless force=True (forcing on an
  ineligible architecture, e.g. one without a value norm, changes model outputs).
  Returns the audit report."""
  layers = decoder_layers(model) if layers is None else layers
  report = audit_pre_attention_cancellation(model, layers)
  bad = [r['layer'] for r in report if not r['eligible']]
  if bad and not force:
    raise ValueError(f'layers {bad} are not eligible for cancellation (missing q/k/v norm); '
                     f'use mla_partial_cancel() for MLA models or force=True to override')
  for ly in layers:
    at = ly.self_attn
    if isinstance(ly.input_layernorm, _BypassedNorm):
      raise ValueError('input_layernorm already cancelled; transforms must be applied once')
    projs = [p for p in (getattr(at, 'q_proj', None), getattr(at, 'k_proj', None),
                         getattr(at, 'v_proj', None)) if p is not None]
    fold_norm_into_projs(ly.input_layernorm, projs, convention)
    ly.input_layernorm = _BypassedNorm(_get_norm_eps(ly.input_layernorm), keep_scalar=False)
  return report


def mla_partial_cancel(model, convention='auto', layers=None, rope_dims=None):
  """Partial cancellation for MLA attention (DeepSeek-V2 family, MiniCPM3).

  Folds the input_layernorm gain into the query and kv_a projections, bypasses the
  full-width normalization multiply, and re-applies the retained per-token RMS scalar
  only where the eligibility criterion fails: the decoupled RoPE-key slice (the last
  rope_dims columns of kv_a_proj_with_mqa's output), plus the whole query projection
  when the model has no q_a_layernorm (e.g. DeepSeek-V2-Lite). The RMS reduction itself
  is kept; the norm weight tensor and the hidden-width multiply are removed.
  Returns a per-layer report of what was folded and where the scalar is re-applied."""
  layers = decoder_layers(model) if layers is None else layers
  if rope_dims is None:
    rope_dims = model.config.qk_rope_head_dim
  report = []
  for i, ly in enumerate(layers):
    at = ly.self_attn
    kv_a = getattr(at, 'kv_a_proj_with_mqa', None)
    assert kv_a is not None and getattr(at, 'kv_a_layernorm', None) is not None, \
        'mla_partial_cancel expects kv_a_proj_with_mqa and kv_a_layernorm'
    if isinstance(ly.input_layernorm, _BypassedNorm):
      raise ValueError('input_layernorm already cancelled; transforms must be applied once')
    q_latent = getattr(at, 'q_a_proj', None)
    q_needs_scalar = q_latent is None or getattr(at, 'q_a_layernorm', None) is None
    q_in = q_latent if q_latent is not None else getattr(at, 'q_proj', None)
    assert q_in is not None, 'could not locate the query input projection'
    fold_norm_into_projs(ly.input_layernorm, [q_in, kv_a], convention)
    keeper = _BypassedNorm(_get_norm_eps(ly.input_layernorm), keep_scalar=True)
    ly.input_layernorm = keeper

    def kpe_hook(mod, inp, out, s=keeper, r=rope_dims):
      out[..., -r:] = out[..., -r:] * s.s
      return out
    kv_a.register_forward_hook(kpe_hook)
    if q_needs_scalar:
      q_in.register_forward_hook(lambda mod, inp, out, s=keeper: out * s.s)
    report.append({'layer': i, 'folded': ['q_a_proj' if q_latent is not None else 'q_proj',
                                          'kv_a_proj_with_mqa'],
                   'rope_slice_scalar': rope_dims, 'q_scalar_hook': q_needs_scalar})
  return report
