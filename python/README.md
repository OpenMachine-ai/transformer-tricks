### Setup
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

### Test FlashNorm
```
python3 test_flashNorm.py
```

### Use the transformer-tricks package
```python
import transformer_tricks.tricks as tt
```
