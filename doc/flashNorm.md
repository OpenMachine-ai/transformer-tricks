<a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a> [![Downloads](https://static.pepy.tech/badge/transformer-tricks)](https://pepy.tech/project/transformer-tricks)

## Setup
```
pip3 install transformer-tricks
```

## Example
The example below converts SmolLM-135M to [FlashNorm](https://arxiv.org/pdf/2407.09577) and measures perplexity of the original and the modified model.
```python
import transformer_tricks as tt

# convert model and store the new model in ./SmolLM-135M_flashNorm_test
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('./SmolLM-135M_flashNorm_test')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('./SmolLM-135M_flashNorm_test', speedup=16)
```
Results:
```
Once upon a time there was a curious little girl
Once upon a time there was a curious little girl
perplexity = 16.083
perplexity = 16.083
```

You can run the example in your browser by clicking on this notebook: <a href="https://colab.research.google.com/github/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_example.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>. Hit "cancel" when it says "Notebook does not have secret access", because we don't need an HF_TOKEN for SmolLM.

TODO: [our HuggingFace repo](https://huggingface.co/open-machine/FlashNorm)

## Test FlashNorm
```shell
# setup
git clone https://github.com/OpenMachine-ai/transformer-tricks.git
pip3 install --quiet -r requirements.txt

# run tests
python3 flashNorm_test.py
```
Results:
```
Once upon a time there was a curious little girl
Once upon a time there was a curious little girl
Once upon a time there was a little girl named
Once upon a time there was a little girl named
perplexity = 16.083
perplexity = 16.083
perplexity = 12.086
perplexity = 12.086
```
To run llama and other LLMs that need an agreement (not SmolLM), you first have to type the following, which will ask for your `hf_token`:
```
huggingface-cli login
```

## Please give us a ⭐ if you like this repo, thanks!
