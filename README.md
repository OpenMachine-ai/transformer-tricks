<h1 align="center"> Transformer Tricks </h1>
<p align="center">
  <a href="https://pepy.tech/projects/transformer-tricks"><img src="https://static.pepy.tech/badge/transformer-tricks" alt="PyPI Downloads"></a>
</p>

---

A collection of tricks to simplify and speed up transformer models:
- Slim attention: [[podcast]](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/slim.pdf), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb)
- Flash normalization: [[podcast]](https://notebooklm.google.com/notebook/cca31378-7f5b-4bfc-a1d7-75c7b279fcb5/audio), [[paper]](https://arxiv.org/abs/2407.09577), [[code]](python)
- Precomputing the first layer: [[podcast]](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio), [[paper]](https://arxiv.org/abs/2402.13388)
- Removing weights from skipless transformers: [[podcast]](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [[paper]](https://arxiv.org/abs/2404.12362), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb)
- Approximate attention [work in progress]: [[podcast]](https://notebooklm.google.com/notebook/5fb65371-6048-4e63-8a37-6e4f16d7f708/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/approximate.pdf)

Many of these tricks follow a recent trend of removing parts from neural networks such as RMSNorm’s removal of mean centering from LayerNorm, T5’s removal of bias-parameters, NoPE’s removal of positional encoding, GPT’s removal of the encoder stack, and of course transformer’s revolutionary removal of recurrent layers. Specifically, our FlashNorm removes the weights from RMSNorm and merges them with the next linear layer. And slim attention removes the entire V-cache from the context memory for MHA transformers.

---

## Getting Started

Install the transformer tricks package with `pip`:
```bash
pip install transformer-tricks
```
---

OLD docu:

Tricks and tools for speeding up LLMs:

* **Slim attention: cut your context memory in half without loss of accuracy [work in progress]:**
  * Notebook for paper:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>

* **Flash normalization:**
  * Notebook example for converting an LLM to FlashNorm: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * Notebook for paper:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flash_normalization.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * [HuggingFace repo](https://huggingface.co/open-machine/FlashNorm)

* **Removing weights from skipless transformers:**
  * Notebook:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>

---

**Please give us a ⭐ if you like this repo, thanks!**
