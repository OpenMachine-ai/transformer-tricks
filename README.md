<h1 align="center"> Transformer Tricks </h1>

<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a> [![Downloads](https://static.pepy.tech/badge/transformer-tricks)](https://pepy.tech/project/transformer-tricks)

```
pip3 install transformer-tricks
```

---

A collection of tricks to speed up and simplify transformer models:
- Slim attention: [[podcast]](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/slim.pdf), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb)
- Flash normalization: [[podcast]](https://notebooklm.google.com/notebook/cca31378-7f5b-4bfc-a1d7-75c7b279fcb5/audio), [[paper]](https://arxiv.org/abs/2407.09577), [[code]](python)
- Precomputing the first layer: [[podcast]](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio), [[paper]](https://arxiv.org/abs/2402.13388)  
- Merging weights for skipless transformers: [[podcast]](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [[paper]](https://arxiv.org/abs/2404.12362), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb)
- Approximate attention [work in progress]: [[podcast]](https://notebooklm.google.com/notebook/5fb65371-6048-4e63-8a37-6e4f16d7f708/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/approximate.pdf)

---
OLD docu:

Tricks and tools for speeding up LLMs:

* **Slim attention: cut your context memory in half without loss of accuracy [work in progress]:**
  * [Podcast](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio)
  * [PDF here](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/slim.pdf)
  * Notebook for paper:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>

* **Flash normalization:**
  * [Podcast](https://notebooklm.google.com/notebook/cca31378-7f5b-4bfc-a1d7-75c7b279fcb5/audio)
  * arXiv paper: https://arxiv.org/abs/2407.09577
  * See [python folder](python) for code to convert LLMs to FlashNorm
  * Notebook example for converting an LLM to FlashNorm: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * Notebook for paper:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flash_normalization.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * [HuggingFace repo](https://huggingface.co/open-machine/FlashNorm)

* **Approximate attention [work in progress]:**
  * [Podcast](https://notebooklm.google.com/notebook/5fb65371-6048-4e63-8a37-6e4f16d7f708/audio)
  * [PDF here](doc/approximate.pdf)

* **Removing weights from skipless transformers:**
  * [Podcast](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio)
  * arXiv paper: https://arxiv.org/abs/2404.12362
  * Notebook:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>

* **Precomputing the first layer:**
  * [Podcast](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio)
  * arXiv paper: https://arxiv.org/abs/2402.13388

---

**Please give us a ⭐ if you like this repo, thanks!**
