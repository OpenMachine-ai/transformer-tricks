<h1 align="center"> Transformer Tricks

  <a href="https://transformertricks.substack.com"><img src="https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff"></a>
  [![PyPI](https://img.shields.io/pypi/v/transformer-tricks)](https://pypi.org/project/transformer-tricks)
  <a href="https://pepy.tech/projects/transformer-tricks"><img src="https://static.pepy.tech/badge/transformer-tricks" alt="PyPI Downloads"></a>
</h1>

A collection of tricks to simplify and speed up transformer models:
- Slim attention: [paper](https://arxiv.org/abs/2503.05840), [video](https://youtu.be/uVtk3B6YO4Y), [podcast](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_paper.ipynb), [code-readme](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/slimAttn.md), :hugs: [article](https://huggingface.co/blog/Kseniase/attentions), [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1j9wkc2/slim_attention_cut_your_context_memory_in_half)
- FlashNorm: [paper](https://arxiv.org/abs/2407.09577), [video](https://youtu.be/GEuJv34_XgU), [podcast](https://notebooklm.google.com/notebook/0877599c-720c-49b5-b451-8a41af592dd1/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_paper.ipynb), [code-readme](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/flashNorm.md)
- MatShrink \[work in progress\]: [paper](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/matShrink.pdf), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/matShrink_paper.ipynb), [precision experiments](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/matShrink_precision.ipynb)
- Precomputing the first layer: [paper](https://arxiv.org/abs/2402.13388), [video](https://youtu.be/pUeSwnCOoNI), [podcast](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio)
- KV-weights only for skipless transformers: [paper](https://arxiv.org/abs/2404.12362), [video](https://youtu.be/Tx_lMpphd2g), [podcast](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removeWeights_paper.ipynb)

These transformer tricks extend a recent trend in neural network design toward architectural parsimony, in which unnecessary components are removed to create more efficient models. Notable examples include [RMSNorm’s](https://arxiv.org/abs/1910.07467) simplification of LayerNorm by removing mean centering, [PaLM's](https://arxiv.org/abs/2204.02311) elimination of bias parameters, and [decoder-only transformer's](https://arxiv.org/abs/1801.10198) omission of the encoder stack. This trend began with the original [transformer model's](https://arxiv.org/abs/1706.03762) removal of recurrence and convolutions.

For example, our [FlashNorm](https://arxiv.org/abs/2407.09577) removes the weights from RMSNorm and merges them with the next linear layer. And [slim attention](https://arxiv.org/abs/2503.05840) removes the entire V-cache from the context memory for MHA transformers.

Transformer tricks GitHub repo: [here](https://github.com/OpenMachine-ai/transformer-tricks)
<!-- Above link is important for the PyPI package, which uses the same README. That way, folks can click from the PyPI website to this repo -->

---

## Explainer videos

[![hey](https://img.youtube.com/vi/uVtk3B6YO4Y/0.jpg)](https://www.youtube.com/watch?v=uVtk3B6YO4Y "Slim attention")
[![hey](https://img.youtube.com/vi/GEuJv34_XgU/0.jpg)](https://www.youtube.com/watch?v=GEuJv34_XgU "Flash normalization")
[![hey](https://img.youtube.com/vi/pUeSwnCOoNI/0.jpg)](https://www.youtube.com/watch?v=pUeSwnCOoNI "Precomputing the first layer")
[![hey](https://img.youtube.com/vi/Tx_lMpphd2g/0.jpg)](https://www.youtube.com/watch?v=Tx_lMpphd2g "Removing weights from skipless transformers")

---

## Installation

Install the transformer tricks package:
```bash
pip install transformer-tricks
```

Alternatively, to run from latest repo:
```bash
git clone https://github.com/OpenMachine-ai/transformer-tricks.git
python3 -m venv .venv
source .venv/bin/activate
pip3 install --quiet -r requirements.txt
```

---

## Flashify your own model

Convert any RMSNorm-based HuggingFace checkpoint to a `-FlashNorm` variant and publish it under your own account. The recipe below does the full round-trip in a dozen lines; it works for Llama, Mistral, Gemma, Qwen, SmolLM, and any other transformer that uses RMSNorm followed by a linear layer.

```python
# pip install transformer-tricks huggingface-hub
import transformer_tricks as tt
from huggingface_hub import HfApi, login

login()                                            # paste your HF write token when prompted

SRC = 'meta-llama/Llama-3.2-1B'                    # source model on HF
OUT = 'YOUR_USERNAME/Llama-3.2-1B-FlashNorm'       # destination (under your account)
LOCAL = './Llama-3.2-1B_flashNorm'                 # local workdir

tt.flashify_repo(SRC, dir=LOCAL, strict=True)      # fold g into W*, remove norm tensors

api = HfApi()
api.create_repo(OUT, exist_ok=True)
api.upload_folder(repo_id=OUT, folder_path=LOCAL)
print(f'Published https://huggingface.co/{OUT}')
```

The `strict=True` flag folds the per-channel norm weights `g` into the following linear layer and removes the now-redundant norm tensors from the state dict entirely. The resulting checkpoint is mathematically equivalent to the source (Proposition 1 of the [FlashNorm paper](https://arxiv.org/abs/2407.09577)). Framework support status (HuggingFace Transformers, vLLM, llama.cpp) is tracked on the canonical FlashNorm checkpoint: [open-machine/SmolLM2-135M-FlashNorm](https://huggingface.co/open-machine/SmolLM2-135M-FlashNorm).

A runnable notebook version of this recipe is at [`notebooks/flashify_and_publish.ipynb`](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashify_and_publish.ipynb).

---

## Cancel redundant norms (Proposition 3)

An RMSNorm followed by a bias-free linear layer followed by another RMSNorm makes the first norm redundant: after folding its gain into the linear layer, scale invariance of the second norm cancels the first norm's division. In QKV-normalized models such as Gemma 4 (which re-normalizes queries, keys, and values per head), this removes the entire pre-attention RMSNorm from every decoder layer.

```python
import flashNorm_cancel as fc  # from a repo checkout; not yet in the PyPI wheel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B')
print(fc.audit_pre_attention_cancellation(model))  # per-layer eligibility report
fc.cancel_pre_attention_norms(model)               # fold gains, remove the division
```

The cancellation is exact for unregularized RMS and epsilon-approximate in practice. Validated on Gemma-4-E2B and Gemma-4-12B in fp32: max logit deviation at the rounding level of the exact weight fold itself, perplexity changes under 0.001%, greedy generations unchanged, HellaSwag unchanged (bf16 spot check).

Important: eligibility must be checked, not assumed. In MLA models (DeepSeek-V2 family, MiniCPM3) the decoupled RoPE-key slice bypasses the latent norm, so full cancellation is invalid there (it breaks the model badly). Use `fc.mla_partial_cancel(model)` instead, which keeps the per-token RMS scalar but applies it only to the small RoPE slice, still eliminating the norm weights and the full-width multiply. Models that normalize only queries and keys (e.g. Gemma 3, no value norm) are not eligible, and `cancel_pre_attention_norms` refuses them by default. Run `python flashNorm_cancel_test.py` for a fast synthetic verification of all these cases.

---

## Documentation
Follow the links below for documentation of the python code in this directory:
- [Slim attention](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/slimAttn.md)
- [Flash normalization](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/flashNorm.md)

---

## Notebooks
The papers are accompanied by the following Jupyter notebooks:
- Slim attention: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Flash normalization: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a> <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a> <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_gpu_benchmark.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Flashify your own model and publish to HuggingFace: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashify_and_publish.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- MatShrink: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/matShrink_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a> <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/matShrink_precision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Removing weights from skipless transformers: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removeWeights_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>

---
## Newsletter
Please subscribe to our [newsletter](https://transformertricks.substack.com) on substack to get the latest news about this project. We will never send you more than one email per month.

[![Substack](https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff)](https://transformertricks.substack.com)

---

## Contributing
We pay cash for high-impact contributions. Please check out [CONTRIBUTING](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/doc/CONTRIBUTING.md) for how to get involved.

---

## Sponsors
The Transformer Tricks project is currently sponsored by [OpenMachine](https://openmachine.ai). We'd love to hear from you if you'd like to join us in supporting this project.

---

### Please give us a ⭐ if you like this repo, and check out [TinyFive](https://github.com/OpenMachine-ai/tinyfive)

---

[![Star History Chart](https://api.star-history.com/svg?repos=OpenMachine-ai/transformer-tricks&type=Date)](https://www.star-history.com/#OpenMachine-ai/transformer-tricks&Date)
