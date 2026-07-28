# Synthetic tests for flashNorm_cancel.py. Runs on CPU in seconds, no model downloads.
# All math in float64 so the only tolerated error is the epsilon approximation itself.
#
# Usage: python flashNorm_cancel_test.py

import copy
import torch
import torch.nn as nn
import flashNorm_cancel as fc

torch.manual_seed(0)
D = torch.float64


class RMS(nn.Module):
  """Minimal RMSNorm with selectable gain convention; scale=False gives a scale-free norm."""

  def __init__(self, dim, eps=1e-6, convention='plain', scale=True):
    super().__init__()
    self.eps = eps
    self.convention = convention
    self.scale = scale
    if scale:
      init = torch.zeros(dim, dtype=D) if convention == 'one_plus' else torch.ones(dim, dtype=D)
      self.weight = nn.Parameter(init + 0.1 * torch.randn(dim, dtype=D))

  def forward(self, x):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    if not self.scale:
      return y
    g = 1.0 + self.weight if self.convention == 'one_plus' else self.weight
    return y * g


class TinyAttn(nn.Module):
  """Gemma-4-like attention block: q/k/v projections with per-head norms after each."""

  def __init__(self, n=16, hd=4, eps=1e-6, convention='plain', v_norm=True, kv_shared=False):
    super().__init__()
    self.hd = hd
    self.q_proj = nn.Linear(n, 2 * hd, bias=False, dtype=D)
    self.q_norm = RMS(hd, eps, convention)
    if kv_shared:
      self.k_proj = None
      self.v_proj = None
      self.k_norm = None
      self.v_norm = None
    else:
      self.k_proj = nn.Linear(n, hd, bias=False, dtype=D)
      self.k_norm = RMS(hd, eps, convention)
      self.v_proj = nn.Linear(n, hd, bias=False, dtype=D)
      self.v_norm = RMS(hd, eps, convention, scale=False) if v_norm else None

  def per_head(self, norm, y):
    return norm(y.view(*y.shape[:-1], -1, self.hd)).view(*y.shape)

  def forward(self, x):
    outs = [self.per_head(self.q_norm, self.q_proj(x))]
    if self.k_proj is not None:
      outs.append(self.per_head(self.k_norm, self.k_proj(x)))
      v = self.v_proj(x)
      outs.append(self.per_head(self.v_norm, v) if self.v_norm is not None else v)
    return torch.cat(outs, -1)


class TinyLayer(nn.Module):
  def __init__(self, n=16, eps=1e-6, convention='plain', **kw):
    super().__init__()
    self.input_layernorm = RMS(n, eps, convention)
    self.self_attn = TinyAttn(n, eps=eps, convention=convention, **kw)

  def forward(self, x):
    return self.self_attn(self.input_layernorm(x))


class TinyMLAAttn(nn.Module):
  """DeepSeek-style MLA block: kv_a splits into a re-normalized latent and a bypassing
  RoPE-key slice; the query path optionally has its own latent norm (q_lora=False mimics
  DeepSeek-V2-Lite, whose query path is un-renormalized)."""

  def __init__(self, n=16, kv_lora=8, rope=4, eps=1e-6, q_lora=True):
    super().__init__()
    self.rope = rope
    if q_lora:
      self.q_a_proj = nn.Linear(n, 8, bias=False, dtype=D)
      self.q_a_layernorm = RMS(8, eps)
      self.q_b_proj = nn.Linear(8, 8, bias=False, dtype=D)
    else:
      self.q_proj = nn.Linear(n, 8, bias=False, dtype=D)
    self.kv_a_proj_with_mqa = nn.Linear(n, kv_lora + rope, bias=False, dtype=D)
    self.kv_a_layernorm = RMS(kv_lora, eps)
    self.kv_b_proj = nn.Linear(kv_lora, 8, bias=False, dtype=D)

  def forward(self, x):
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))) if hasattr(self, 'q_a_proj') \
        else self.q_proj(x)
    kv = self.kv_a_proj_with_mqa(x)
    latent, k_pe = kv[..., :-self.rope], kv[..., -self.rope:]
    return torch.cat([q, self.kv_b_proj(self.kv_a_layernorm(latent)), k_pe], -1)


class TinyMLALayer(nn.Module):
  def __init__(self, eps=1e-6, q_lora=True):
    super().__init__()
    self.input_layernorm = RMS(16, eps)
    self.self_attn = TinyMLAAttn(eps=eps, q_lora=q_lora)

  def forward(self, x):
    return self.self_attn(self.input_layernorm(x))


def max_rel(a, b):
  return ((a - b).abs().max() / b.abs().max()).item()


def check(name, err, tol):
  status = 'PASS' if err <= tol else 'FAIL'
  print(f'{status}  {name}: max rel err {err:.3e} (tol {tol:.0e})')
  assert err <= tol, name


def main():
  x = torch.randn(3, 5, 16, dtype=D)

  # 1. eligible Gemma-4-like layer: cancellation matches the reference within epsilon effects
  for eps, tol in ((0.0, 1e-12), (1e-6, 5e-5)):
    ref = TinyLayer(eps=eps).double()
    out_ref = ref(x)
    m = copy.deepcopy(ref)
    report = fc.cancel_pre_attention_norms(None, convention='plain', layers=[m])
    assert all(r['eligible'] for r in report)
    check(f'gemma4-like cancel, eps={eps}', max_rel(m(x), out_ref), tol)

  # 2. KV-shared layer (no k/v projections): still eligible, q path only
  ref = TinyLayer(kv_shared=True).double()
  out_ref = ref(x)
  m = copy.deepcopy(ref)
  fc.cancel_pre_attention_norms(None, convention='plain', layers=[m])
  check('kv-shared cancel', max_rel(m(x), out_ref), 5e-5)

  # 3. one_plus gain convention: auto-detected and folded correctly (exact at eps=0)
  for eps, tol in ((0.0, 1e-12), (1e-6, 5e-5)):
    ref = TinyLayer(convention='one_plus', eps=eps).double()
    assert fc.detect_gain_convention(ref.input_layernorm) == 'one_plus'
    out_ref = ref(x)
    m = copy.deepcopy(ref)
    fc.cancel_pre_attention_norms(None, convention='auto', layers=[m])
    check(f'one_plus convention cancel, eps={eps}', max_rel(m(x), out_ref), tol)

  # 4. negative control: Gemma-3-like (no v_norm) must refuse, and forcing must change outputs
  ref = TinyLayer(v_norm=False).double()
  m = copy.deepcopy(ref)
  try:
    fc.cancel_pre_attention_norms(None, convention='plain', layers=[m])
    raise AssertionError('ineligible layer was not refused')
  except ValueError:
    print('PASS  ineligible layer refused without force=True')
  m = copy.deepcopy(ref)
  fc.cancel_pre_attention_norms(None, convention='plain', layers=[m], force=True)
  err = max_rel(m(3.0 * x), ref(3.0 * x))
  print(f'PASS  forced ineligible cancel diverges as expected: max rel err {err:.3e}')
  assert err > 1e-3, 'forced cancel on ineligible layer should change outputs'

  # 4b. MLA layer must be refused by full cancellation (audit must not pass vacuously)
  try:
    fc.cancel_pre_attention_norms(None, convention='plain', layers=[TinyMLALayer().double()])
    raise AssertionError('MLA layer was not refused by cancel_pre_attention_norms')
  except ValueError:
    print('PASS  MLA layer refused by full cancellation')

  # 4c. fold-only with the norm still executing: identity weight must be correct per convention
  for conv in ('plain', 'one_plus'):
    ref = TinyLayer(convention=conv).double()
    out_ref = ref(x)
    m = copy.deepcopy(ref)
    fc.fold_norm_into_projs(m.input_layernorm,
                            [m.self_attn.q_proj, m.self_attn.k_proj, m.self_attn.v_proj],
                            convention=conv)
    check(f'fold-only, norm live, {conv}', max_rel(m(x), out_ref), 1e-12)

  # 4d. two-layer cancel: hook/closure binding must stay per-layer
  refs = [TinyLayer().double() for _ in range(2)]
  outs = [r(x) for r in refs]
  ms = [copy.deepcopy(r) for r in refs]
  fc.cancel_pre_attention_norms(None, convention='plain', layers=ms)
  for i in range(2):
    check(f'two-layer cancel, layer {i}', max_rel(ms[i](x), outs[i]), 5e-5)

  # 5. MLA partial cancellation: matches reference within epsilon effects; naive bypass diverges
  for q_lora, tag in ((True, 'deepseek-v2-like'), (False, 'v2-lite-like (bare query path)')):
    ref = TinyMLALayer(q_lora=q_lora).double()
    out_ref = ref(x)
    m = copy.deepcopy(ref)
    fc.mla_partial_cancel(None, convention='plain', layers=[m], rope_dims=4)
    check(f'mla partial cancel, {tag}', max_rel(m(x), out_ref), 5e-5)
    x2 = torch.randn(3, 5, 16, dtype=D)  # fresh input: catches a keeper caching stale s
    check(f'mla partial cancel, {tag}, second forward on fresh input',
          max_rel(m(x2), ref(x2)), 5e-5)
    naive = copy.deepcopy(ref)
    fc.fold_norm_into_projs(naive.input_layernorm,
                            [naive.self_attn.q_a_proj if q_lora else naive.self_attn.q_proj,
                             naive.self_attn.kv_a_proj_with_mqa], convention='plain')
    naive.input_layernorm = fc._RmsScalarKeeper(1e-6, keep_scalar=False)
    err = max_rel(naive(3.0 * x), ref(3.0 * x))
    print(f'PASS  naive MLA bypass diverges as expected ({tag}): max rel err {err:.3e}')
    assert err > 1e-3, 'naive MLA bypass should change outputs'

  print('\nall tests passed')


if __name__ == '__main__':
  main()
