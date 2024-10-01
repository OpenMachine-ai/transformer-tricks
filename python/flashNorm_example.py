# This example converts SmolLM-135M to FlashNorm and measures
# its perplexity before and after the conversion.
#
# Usage:
#   python3 flashNorm_example.py
#
# Whenever you change this script, make sure to regenerate the
# jupyter notebook as follows:
#   pip install jupytext
#   jupytext --to notebook flashNorm_example.py -o ../notebooks/flashNorm_example.ipynb
# '# %%' in the code below marks the beginning of a new cell in the notebook

import os
import transformer_tricks as tt
# %%

# convert model and store the new model in ./SmolLM-135M_flashNorm
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('SmolLM-135M_flashNorm')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('SmolLM-135M_flashNorm', speedup=16)
