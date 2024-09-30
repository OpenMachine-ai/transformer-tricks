## Setup
```
pip3 install transformer-tricks
```
To run llama and other LLMs that need an agreement (not SmolLM), you first have to type the following, which will ask for your `hf_token`:
```
huggingface-cli login
```

## Example
The example below converts SmolLM-135M to FlashNorm and measures perplexity of the original and the modified model.
```python
import transformer_tricks as tt

# convert model and store the new model in ./SmolLM-135M_flashNorm
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('SmolLM-135M_flashNorm')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('SmolLM-135M_flashNorm', speedup=16)
```
Results:
```
Once upon a time there was a curious little girl
Once upon a time there was a curious little girl
ppl: tensor(16.0831)
ppl: tensor(16.0831)
```

## Test FlashNorm
```
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
ppl: tensor(16.0831)
ppl: tensor(16.0831)
ppl: tensor(12.0864)
ppl: tensor(12.0864)
```

## Notes on python package
- Link to package [here](https://pypi.org/project/transformer-tricks/)
- Link to stats [here](https://www.pepy.tech/projects/transformer-tricks)
