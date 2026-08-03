# Precision study for "Slim Attention" on Whisper cross-attention.
# Usage: python3 slimAttn_whisper.py [model]    per-layer detail for one model
#        python3 slimAttn_whisper.py --sweep    which option survives which dtype
#
# Slim attention caches one of K or V and recomputes the other:
#   Option 1: cache K, V = K @ inv(Wk) @ Wv
#   Option 2: cache V, K = V @ inv(Wv) @ Wk
# Both halve the cache. This measures which one survives fp16 per layer, using
# real encoder activations rather than random keys, because the two options
# respond to real speech in opposite directions.
#
# --sweep adds Option 3 (cache X, recompute both K and V). It needs no inverse,
# so it has no conditioning to lose, and it is the only option that survives
# bf16. Option 1 turns out never to survive fp16 at any model size.

# %pip install --quiet transformer_tricks datasets soundfile
import io
import sys
import torch
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

FP16_MAX = 65504.0  # largest finite fp16; W above this is not representable
BAD = 0.05          # relative error we treat as a failed reconstruction

#-------------------------------------------------------------------------------
# defs
#-------------------------------------------------------------------------------


def real_activations(model, fe):
  """Encoder output for one real utterance, float64. Random keys understate the
  error for Option 1 and overstate it for Option 2, so use real speech."""
  ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
  ds = ds.cast_column('audio', Audio(decode=False))  # decode ourselves, no torchcodec
  raw = ds[0]['audio']
  wav, _ = sf.read(io.BytesIO(raw['bytes']) if raw.get('bytes') else raw['path'], dtype='float32')
  feats = fe(wav, sampling_rate=16000, return_tensors='pt').input_features
  with torch.no_grad():
    return model.model.encoder(feats).last_hidden_state[0].double()


def reconstruct(A, B, X):
  """Cache X@A.T, recompute X@B.T through W = inv(A) @ B built in float64.
  Returns max|W| and the fp16 relative error."""
  W = torch.linalg.solve(A.T, B.T)  # more stable than forming inv(A) explicitly
  src, dst = X @ A.T, X @ B.T
  got = (src.to(torch.float16) @ W.to(torch.float16)).double()
  return W.abs().max().item(), ((got - dst).norm() / dst.norm()).item()


def head_residual(Wk, Wv, heads):
  """Can the choice be made per head? Only if rowspace(Wv[h]) sits inside
  rowspace(Wk[h]). Returns the mean least-squares residual over heads, where
  0 means per-head is exact and 1 means the subspaces do not overlap."""
  dh = Wk.shape[0] // heads
  res = []
  for h in range(heads):
    A, B = Wk[h * dh:(h + 1) * dh, :].T, Wv[h * dh:(h + 1) * dh, :].T
    res.append(((A @ torch.linalg.lstsq(A, B).solution - B).norm() / B.norm()).item())
  return sum(res) / len(res)


def sweep(models=('tiny', 'base', 'small', 'medium', 'large-v3'), bad=0.05):
  """Layers under `bad` relative error, per option per dtype, per model."""
  dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
  print(f'layers under {bad:.0%} relative error, X from a real encoder pass\n')
  print(f'{"model":>9} {"dtype":>6} {"opt1":>7} {"opt2":>7} {"opt3":>7}')
  for name in models:
    m = WhisperForConditionalGeneration.from_pretrained(
        f'openai/whisper-{name}', dtype=torch.float32).eval()
    fe = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{name}')
    X = real_activations(m, fe)
    attn = [mod for n_, mod in m.named_modules() if n_.endswith('encoder_attn')]
    for dname, dt in dtypes.items():
      ok = [0, 0, 0]
      for mod in attn:
        Wk = mod.k_proj.weight.detach().double()
        Wv = mod.v_proj.weight.detach().double()
        K, V = X @ Wk.T, X @ Wv.T
        e1 = ((K.to(dt) @ torch.linalg.solve(Wk.T, Wv.T).to(dt)).double() - V).norm() / V.norm()
        e2 = ((V.to(dt) @ torch.linalg.solve(Wv.T, Wk.T).to(dt)).double() - K).norm() / K.norm()
        Xd = X.to(dt)   # option 3: no inverse, just the two original projections
        e3 = max(((Xd @ Wk.T.to(dt)).double() - K).norm() / K.norm(),
                 ((Xd @ Wv.T.to(dt)).double() - V).norm() / V.norm())
        for i, e in enumerate((e1, e2, e3)):
          ok[i] += e < bad
      n_ = len(attn)
      print(f'{name:>9} {dname:>6} {ok[0]:>4}/{n_:<2} {ok[1]:>4}/{n_:<2} {ok[2]:>4}/{n_:<2}')
    print()


if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
  sweep()
  sys.exit()

#-------------------------------------------------------------------------------
# measure each cross-attention layer
#-------------------------------------------------------------------------------
name = sys.argv[1] if len(sys.argv) > 1 else 'small'
# large-v3 ships fp16 weights while the feature extractor emits float32, so
# load everything float32 and let the math below promote to float64.
model = WhisperForConditionalGeneration.from_pretrained(
    f'openai/whisper-{name}', dtype=torch.float32).eval()
fe = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{name}')

X = real_activations(model, fe)
Xrand = torch.randn_like(X) * X.std() + X.mean()  # matched mean and std

print(f'whisper-{name}: {tuple(X.shape)} real encoder activations\n')
print(f'{"layer":>5} {"cond(Wk)":>9} {"max|Wkv|":>9} {"opt1":>7} {"opt1rnd":>8} '
      f'{"cond(Wv)":>9} {"max|Wvk|":>9} {"opt2":>7} {"opt2rnd":>8}  {"use":>7} {"head":>5}')

rescued = neither = total = 0
for mod_name, mod in model.named_modules():
  if not mod_name.endswith('encoder_attn'):
    continue
  total += 1
  Wk = mod.k_proj.weight.detach().double()
  Wv = mod.v_proj.weight.detach().double()

  w1, e1 = reconstruct(Wk, Wv, X)       # Option 1: cache K, recompute V
  w2, e2 = reconstruct(Wv, Wk, X)       # Option 2: cache V, recompute K
  _, e1r = reconstruct(Wk, Wv, Xrand)
  _, e2r = reconstruct(Wv, Wk, Xrand)

  # a NaN error means W overflowed fp16, so treat it as failed
  bad1 = w1 > FP16_MAX or not e1 < BAD
  bad2 = w2 > FP16_MAX or not e2 < BAD
  use = 'opt1' if not bad1 else ('opt2' if not bad2 else 'neither')
  rescued += bad1 and not bad2
  neither += bad1 and bad2

  layer = mod_name.replace('model.decoder.layers.', '').replace('.encoder_attn', '')
  print(f'{layer:>5} {torch.linalg.cond(Wk):9.1e} {w1:9.1e} {e1:7.3f} {e1r:8.3f} '
        f'{torch.linalg.cond(Wv):9.1e} {w2:9.1e} {e2:7.3f} {e2r:8.3f}  {use:>7} '
        f'{head_residual(Wk, Wv, mod.num_heads):5.2f}')

print(f'\nOption 2 rescues {rescued}/{total} layers that Option 1 loses; '
      f'{neither}/{total} survive neither.')
print('opt1 is worse on real speech than on random keys, opt2 is better, so the')
print('per-layer choice has to be calibrated on real activations.')
print('The head column is the per-head least-squares residual: near 1 everywhere,')
print('so Wkv mixes heads and the choice cannot be made per head.')
