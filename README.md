<h1 align="center"> Transformer Tricks

  <a href="https://transformertricks.substack.com"><img src="https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff"></a>
  [![PyPI](https://img.shields.io/pypi/v/transformer-tricks)](https://pypi.org/project/transformer-tricks)
  <a href="https://pepy.tech/projects/transformer-tricks"><img src="https://static.pepy.tech/badge/transformer-tricks" alt="PyPI Downloads"></a>
</h1>

A collection of tricks to simplify and speed up transformer models:
- Slim attention: [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/slimAttn.pdf), [[video]](https://youtu.be/uVtk3B6YO4Y), [[podcast]](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_paper.ipynb), [[code-readme]](doc/slimAttn.md)
- Matrix-shrink \[work in progress\]: [[paper]](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/matShrink.pdf)
- Flash normalization: [[paper]](https://arxiv.org/abs/2407.09577), [[podcast]](https://notebooklm.google.com/notebook/0877599c-720c-49b5-b451-8a41af592dd1/audio), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_paper.ipynb), [[code-readme]](doc/flashNorm.md)
- Precomputing the first layer: [[paper]](https://arxiv.org/abs/2402.13388), [[podcast]](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio)
- Removing weights from skipless transformers: [[paper]](https://arxiv.org/abs/2404.12362), [[podcast]](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [[notebook]](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removeWeights_paper.ipynb)

Many of these tricks follow a recent trend of removing parts from neural networks such as RMSNorm’s removal of mean centering from LayerNorm, PaLM's removal of bias-parameters, NoPE’s removal of positional encoding, GPT’s removal of the encoder stack, and of course transformer’s revolutionary removal of recurrent layers. Specifically, our FlashNorm removes the weights from RMSNorm and merges them with the next linear layer. And slim attention removes the entire V-cache from the context memory for MHA transformers.

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
- Slim attention: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Flash normalization: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a> <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>
- Removing weights from skipless transformers: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removeWeights_paper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"></a>

---
## Newsletter
Please subsribe to our [[newsletter]](https://transformertricks.substack.com) on substack to get the latest news about this project. We will never send you more than one email per month.

[![Substack](https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff)](https://transformertricks.substack.com)

---

## Contributing
Before making a change to this repo, please do the following:
- Format your code by typing `autopep8 *.py`. It's using the config in  `pyproject.toml`
- Whenever you change `transformer_tricks.py`, publish a new version of the package as follows:
  - First, update the version number in `pyproject.toml` and in `requirements.txt`
  - Then, push the package to PyPi by typing `./push_pypi.sh`
  - Links for python package: [pypi](https://pypi.org/project/transformer-tricks/), [stats](https://www.pepy.tech/projects/transformer-tricks), [source of this readme](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/README.md)
- Generate notebooks from python by typing `util/gen_notebooks`

---

<h3 align="center"> Please give us a ⭐ if you like this repo, thanks! </h3>

---
