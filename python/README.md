## Setup
To use the latest transformer-tricks python package:
```
pip3 install transformer-tricks
```

If you want to use the latest version of `tricks.py`, do this:
```
pip3 install --quiet -r requirements.txt
```

To run llama and other LLMs that need an agreement (not SmolLM), you first have to type the following:
```
huggingface-cli login
```
Above will ask you for the hf_token, which is the same you use e.g. in colab

## Test FlashNorm
```
python3 test_flashNorm.py
```
Above should return the following:
```
Once upon a time there was a curious little girl
Once upon a time there was a curious little girl
Once upon a time there was a little girl named
Once upon a time there was a little girl named
ppl: tensor(16.0831)
ppl: tensor(16.0831)
ppl: tensor(12.0864)
ppl: tensor(12.0864)
```

## Use the transformer-tricks package
```python
import transformer_tricks as tt
```
## Example
Below example converts the model SmolLM-135M to FlashNorm and measures perplexity of the original and the modified model.
```python
import transformer_tricks as tt

# convert model to flashNorm
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('SmolLM-135M_flashNorm')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('SmolLM-135M_flashNorm', speedup=16)
```
Above should return the following:
```
Once upon a time there was a curious little girl
Once upon a time there was a curious little girl
ppl: tensor(16.0831)
ppl: tensor(16.0831)
```

## Notes on python package
- Link to package [here](https://pypi.org/project/transformer-tricks/)
- Link to stats [here](https://www.pepy.tech/projects/transformer-tricks)
