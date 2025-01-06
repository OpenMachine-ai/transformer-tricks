<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a> [![Downloads](https://static.pepy.tech/badge/transformer-tricks)](https://pepy.tech/project/transformer-tricks)

```
pip3 install transformer-tricks
```
Tricks and tools for speeding up LLMs:

* **Flash normalization:**
  * arXiv paper: https://arxiv.org/abs/2407.09577
  * See [python folder](python) for code to convert LLMs to FlashNorm
  * Notebook example for converting an LLM to FlashNorm: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * Notebook for paper:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flash_normalization.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  * [HuggingFace repo](https://huggingface.co/open-machine/FlashNorm)

* **Slim attention: cut your context memory in half without loss of accuracy [work in progress]:**
  * [PDF here](pdf/slim.pdf)
  * [test](https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/pdf/slim.pdf)

* **Approximate attention [work in progress]:**
  * [PDF here](pdf/approximate.pdf)

* **Removing weights for skipless transformers:**
  * arXiv paper: https://arxiv.org/abs/2404.12362
  * Notebook:
<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>

* **Precomputing the first layer:**
  * arXiv paper: https://arxiv.org/abs/2402.13388

**Please give us a ⭐ if you like this repo, thanks!**
