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

## Test FlashNorm
```shell
# setup
git clone https://github.com/OpenMachine-ai/transformer-tricks.git
cd python
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

## Contributing
Before making a change to this repo, please do the following:
- Format your code by typing `autopep8 *.py`. It's using the config in  `pyproject.toml`.
- Whenever you change `transformer_tricks.py`, publish a new version of the package as follows:
  - First, update the version number in `pyproject.toml` and in `requirements.txt`
  - Then, push the package to PyPi by typing `./push_pypi.sh`
- Whenever you modify `flashNorm_example.py`, generate the corresponding notebook as follows:
  ```
  jupytext --to ipynb flashNorm_example.py -o ../notebooks/flashNorm_example.ipynb
  sed -i -e 's/import transformer_tricks/%pip install --quiet transformer_tricks\\n", "import transformer_tricks/g'
    ../notebooks/flashNorm_example.ipynb
  ```

## Notes on python package
- Link to package [here](https://pypi.org/project/transformer-tricks/)
- Link to stats [here](https://www.pepy.tech/projects/transformer-tricks)
- Source of this README file [here](https://github.com/OpenMachine-ai/transformer-tricks/blob/main/python/README.md)

## Please give us a ⭐ if you like this repo, thanks!
