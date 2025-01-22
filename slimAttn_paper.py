# Proof of concept for paper "Slim Attention: cut your context memory in half"
# Usage: python3 slimAttn_paper.py

# %pip install --quiet transformer_tricks
import transformer_tricks as tt
import numpy as np
import torch
from transformers import AutoConfig

#-------------------------------------------------------------------------------
# defs
#-------------------------------------------------------------------------------
def softmax(x, axis=-1):
  """softmax along 'axis', default is the last axis"""
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / np.sum(e_x, axis=axis, keepdims=True)

def msplit(M, h):
  """shortcut to split matrix M into h chunks"""
  return np.array_split(M, h, axis=-1)

def ops(A, B):
  """number of OPs (operations) for matmul of A and B:
   - A and B must be 2D arrays, and their inner dimensions must agree!
   - A is an m × n matrix, and B is an n × p matrix, then the resulting product
     of A and B is an m × p matrix.
   - Each element (i,j) of the m x p result matrix is computed by the dotproduct
     of the i-th row of A and the j-th column of B.
   - Each dotproduct takes n multiplications and n - 1 additions, so total
     number of OPs is 2n - 1 per dotproduct.
   - There are m * p elements in the result matrix, so m * p dotproducts, so in
     total we need m * p * (2n - 1) OPs, which is approximately 2*m*p*n OPs
   - For simplicity, let's just use the simple approximation of OPs = 2*m*p*n"""
  m, n = A.shape
  p = B.shape[1]
  return 2 * m * n * p

#-------------------------------------------------------------------------------
# setup for model SmolLM2-1.7B
#-------------------------------------------------------------------------------
tt.quiet_hf()  # calm down HuggingFace

repo = 'HuggingFaceTB/SmolLM2-1.7B'
param = tt.get_param(repo)
config = AutoConfig.from_pretrained(repo)

h = config.num_attention_heads
d = config.hidden_size
dk = config.head_dim

# %%
#-------------------------------------------------------------------------------
# check if we can accurately compute V from K for each layer
#-------------------------------------------------------------------------------
for layer in range(config.num_hidden_layers):
  # convert to float64 for better accuracy of matrix inversion
  # note that all weights are transposed in tensorfile (per pytorch convention)
  Wk = param[tt.weight('K', layer)].to(torch.float64).numpy().T
  Wv = param[tt.weight('V', layer)].to(torch.float64).numpy().T
  Wkv = np.linalg.inv(Wk) @ Wv
  print(layer, ':', np.allclose(Wk @ Wkv, Wv))  # check if Wk @ Wkv close to Wv

# %%
#-------------------------------------------------------------------------------
# compare options 1 and 2 for calculating equation (5) of paper
#-------------------------------------------------------------------------------

# get weights for Q, K, V and convert to float64
# note that all weights are transposed in tensorfile (per pytorch convention)
Wq = param[tt.weight('Q', 0)].to(torch.float64).numpy().T
Wk = param[tt.weight('K', 0)].to(torch.float64).numpy().T
Wv = param[tt.weight('V', 0)].to(torch.float64).numpy().T
Wkv = np.linalg.inv(Wk) @ Wv  # calculate Wkv (aka W_KV)
# print('Is Wk @ Wkv close to Wv?', np.allclose(Wk @ Wkv, Wv))

# generate random input X
n = 100  # number of tokens
X = np.random.rand(n, d).astype(np.float64)  # range [0,1]
Xn = np.expand_dims(X[n-1, :], axis=0)  # n-th row of X; make it a 1 x d matrix

Q = Xn @ Wq  # only for the last row of X (for the generate-phase)
K = X @ Wk
V = X @ Wv

# only consider the first head
Q0, K0, V0 = msplit(Q, h)[0], msplit(K, h)[0], msplit(V, h)[0]
Wkv0 = msplit(Wkv, h)[0]

# baseline reference
scores = softmax((Q0 @ K0.T) / np.sqrt(dk))
head_ref = scores @ V0

# head option1 and option2
head_o1 = scores @ (K @ Wkv0)  # option 1
head_o2 = (scores @ K) @ Wkv0  # option 2

# compare
print('Is head_o1 close to head_ref?', np.allclose(head_o1, head_ref))
print('Is head_o2 close to head_ref?', np.allclose(head_o2, head_ref))

# computational complexity for both options
o1_step1, o1_step2 = ops(K, Wkv0), ops(scores, (K @ Wkv0))
o2_step1, o2_step2 = ops(scores, K), ops(scores @ K, Wkv0)

print(f'Option 1 OPs: step 1 = {o1_step1:,}; step 2 = {o1_step2:,}; total = {(o1_step1 + o1_step2):,}')
print(f'Option 2 OPs: step 1 = {o2_step1:,}; step 2 = {o2_step2:,}; total = {(o2_step1 + o2_step2):,}')
print(f'speedup of option 2 over option 1: {((o1_step1 + o1_step2) / (o2_step1 + o2_step2)):.1f}')

# %% [markdown]
# Whenever you change this file, make sure to regenerate the jupyter notebook by typing:
#   `util/gen_notebooks`
