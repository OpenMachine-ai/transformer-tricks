<h1 align="center"> Transformer Tricks </h1>
<p align="center">
  <a href="https://transformertricks.substack.com"><img src="https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff"></a>
  <a href="https://pepy.tech/projects/transformer-tricks"><img src="https://static.pepy.tech/badge/transformer-tricks" alt="PyPI Downloads"></a>
</p>

---

A collection of tricks to simplify and speed up transformer models:
- Slim attention: [[podcast]](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/slim.pdf), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb), [[code-readme]](doc/slimAttn.md)
- Flash normalization: [[podcast]](https://notebooklm.google.com/notebook/cca31378-7f5b-4bfc-a1d7-75c7b279fcb5/audio), [[paper]](https://arxiv.org/abs/2407.09577), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flash_normalization.ipynb), [[code-readme]](doc/flashNorm.md)
- Precomputing the first layer: [[podcast]](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio), [[paper]](https://arxiv.org/abs/2402.13388)
- Removing weights from skipless transformers: [[podcast]](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [[paper]](https://arxiv.org/abs/2404.12362), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb)
- Approximate attention [work in progress]: [[podcast]](https://notebooklm.google.com/notebook/5fb65371-6048-4e63-8a37-6e4f16d7f708/audio), [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/approximate.pdf)

Many of these tricks follow a recent trend of removing parts from neural networks such as RMSNorm’s removal of mean centering from LayerNorm, T5’s removal of bias-parameters, NoPE’s removal of positional encoding, GPT’s removal of the encoder stack, and of course transformer’s revolutionary removal of recurrent layers. Specifically, our FlashNorm removes the weights from RMSNorm and merges them with the next linear layer. And slim attention removes the entire V-cache from the context memory for MHA transformers.

---

## Installation

Install the transformer tricks package:
```bash
pip install transformer-tricks
```

---

## Documentation
Follow the links below for documentation of the python code in this directory:
- [Slim attention](doc/slimAttn.md)
- [Flash normalization](doc/flashNorm.md)

---

## Notebooks
The papers are accompanied by the following Jupyter notebooks:
- Slim attention: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_concept.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Flash normalization: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a> <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flash_normalization.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Removing weights from skipless transformers: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removing_weights.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>

---
## Newsletter
Please subsribe to our [[newsletter]](https://transformertricks.substack.com) on substack to get the latest news about this project. We will never send you more than one email per month.

[![Substack](https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff)](https://transformertricks.substack.com)

---

## Contributing
Before making a change to this repo, please do the following:
- Format your code by typing `autopep8 *.py`. It's using the config in  `pyproject.toml`.
- Whenever you change `transformer_tricks.py`, publish a new version of the package as follows:
  - First, update the version number in `pyproject.toml` and in `requirements.txt`
  - Then, push the package to PyPi by typing `./push_pypi.sh`
  - Links for python package: [pypi](https://pypi.org/project/transformer-tricks/), [stats](https://www.pepy.tech/projects/transformer-tricks), [source of this readme](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/python/README.md)
- Whenever you modify `flashNorm_example.py` or another python file, generate the corresponding notebook as follows:
  ```
  jupytext --to ipynb flashNorm_example.py -o notebooks/flashNorm_example.ipynb
  ```

---

<h3 align="center"> Please give us a ⭐ if you like this repo, thanks! </h3>

---
