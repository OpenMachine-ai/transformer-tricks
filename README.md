<h1 align="center"> Transformer Tricks

  <a href="https://transformertricks.substack.com"><img src="https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff"></a>
  [![PyPI](https://img.shields.io/pypi/v/transformer-tricks)](https://pypi.org/project/transformer-tricks)
  <a href="https://pepy.tech/projects/transformer-tricks"><img src="https://static.pepy.tech/badge/transformer-tricks" alt="PyPI Downloads"></a>
</h1>

A collection of tricks to simplify and speed up transformer models:
- Slim attention: [paper](https://arxiv.org/abs/2503.05840), [video](https://youtu.be/uVtk3B6YO4Y), [podcast](https://notebooklm.google.com/notebook/ac47a53c-866b-4271-ab79-bc48d1b41722/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/slimAttn_paper.ipynb), [code-readme](doc/slimAttn.md), :hugs: [article](https://huggingface.co/blog/Kseniase/attentions), [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1j9wkc2/slim_attention_cut_your_context_memory_in_half)
- Flash normalization: [paper](https://arxiv.org/abs/2407.09577), [video](https://youtu.be/GEuJv34_XgU), [podcast](https://notebooklm.google.com/notebook/0877599c-720c-49b5-b451-8a41af592dd1/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_paper.ipynb), [code-readme](doc/flashNorm.md)
- Matrix-shrink \[work in progress\]: [paper](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/doc/matShrink.pdf)
- Precomputing the first layer: [paper](https://arxiv.org/abs/2402.13388), [video](https://youtu.be/pUeSwnCOoNI), [podcast](https://notebooklm.google.com/notebook/7794278e-de6a-40fc-ab1c-3240a40e55d5/audio)
- Removing weights from skipless transformers: [paper](https://arxiv.org/abs/2404.12362), [video](https://youtu.be/Tx_lMpphd2g), [podcast](https://notebooklm.google.com/notebook/0875eef7-094e-4c30-bc13-90a1a074c949/audio), [notebook](https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/removeWeights_paper.ipynb)

Many of these tricks follow a recent trend of removing parts from neural networks such as [RMSNorm’s](https://arxiv.org/abs/1910.07467) removal of mean centering from LayerNorm, [PaLM's](https://arxiv.org/abs/2204.02311) removal of bias-parameters, [decoder-only transformer's](https://arxiv.org/abs/1801.10198) removal of the encoder stack, and of course [transformer’s](https://arxiv.org/abs/1706.03762) revolutionary removal of recurrence and convolutions.

For example, our FlashNorm removes the weights from RMSNorm and merges them with the next linear layer. And slim attention removes the entire V-cache from the context memory for MHA transformers.

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
Please subscribe to our [newsletter](https://transformertricks.substack.com) on substack to get the latest news about this project. We will never send you more than one email per month.

[![Substack](https://img.shields.io/badge/Substack-FF6719?logo=substack&logoColor=fff)](https://transformertricks.substack.com)

---

## Contributing
We pay cash for high-impact contributions. Please check out [CONTRIBUTING](doc/CONTRIBUTING.md) for how to get involved.

---

## Sponsors
The Transformer Tricks project is currently sponsored by [OpenMachine](https://openmachine.ai). We'd love to hear from you if you'd like to join us in supporting this project.

---

### Please give us a ⭐ if you like this repo, and check out [TinyFive](https://github.com/OpenMachine-ai/tinyfive)

---

[![Star History Chart](https://api.star-history.com/svg?repos=OpenMachine-ai/transformer-tricks&type=Date)](https://www.star-history.com/#OpenMachine-ai/transformer-tricks&Date)
