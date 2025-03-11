# This example converts SmolLM-135M to FlashNorm and measures its perplexity
# before and after the conversion.
# Usage: python3 flashNorm_example.py

# !wget -q https://raw.githubusercontent.com/OpenMachine-ai/transformer-tricks/refs/heads/main/flashNorm_modeling_llama.py
# %pip install --quiet transformer_tricks
import transformer_tricks as tt

tt.quiet_hf()  # calm down HuggingFace

# %%
#-------------------------------------------------------------------------------
# Example 1
#-------------------------------------------------------------------------------

# convert model and store the new model in ./SmolLM-135M_flashNorm_test
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('./SmolLM-135M_flashNorm_test')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('./SmolLM-135M_flashNorm_test', speedup=16)

# %%
#-------------------------------------------------------------------------------
# Example 2
#-------------------------------------------------------------------------------

# convert model and store the new model in ./SmolLM-135M_flashNorm
tt.flashify_repo('HuggingFaceTB/SmolLM-135M')

# run example inference of original and modified model
tt.hello_world('HuggingFaceTB/SmolLM-135M')
tt.hello_world('./SmolLM-135M_flashNorm', arch='LlamaFlashNorm')

# measure perplexity of original and modified model
tt.perplexity('HuggingFaceTB/SmolLM-135M', speedup=16)
tt.perplexity('./SmolLM-135M_flashNorm', speedup=16, arch='LlamaFlashNorm')

# %% [markdown]
# Whenever you change this file, make sure to regenerate the jupyter notebook by typing:
#   `util/gen_notebooks`
