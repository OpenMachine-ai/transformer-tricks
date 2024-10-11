# This example converts SmolLM-135M to FlashNorm and measures
# its perplexity before and after the conversion.
#
# Usage:
#   python3 flashNorm_example.py

import transformer_tricks as tt

tt.quiet_hf()  # calm down HuggingFace

# convert model and store the new model in ./SmolLM-135M_flashNorm
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('./SmolLM-135M_flashNorm')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('./SmolLM-135M_flashNorm', speedup=16)

# %% [markdown]
# Whenever you change this file, make sure to regenerate the jupyter notebook as follows:
#   jupytext --to ipynb flashNorm_example.py -o ../notebooks/flashNorm_example.ipynb
#   sed -i -e 's/import transformer_tricks/%pip install --quiet transformer_tricks\\n", "import transformer_tricks/g'
#     ../notebooks/flashNorm_example.ipynb
#   The sed command adds %pip install --quiet transformer-tricks
