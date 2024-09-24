# calculate perplexity (PPL) of a model with wikitext2
#
# this file is copied from https://huggingface.co/docs/transformers/perplexity
# I made the following changes to adapt it for SmolLM (was GPT2 before):
#   - changed 'model' and 'tokenizer'
#   - changed 'from transformers import' to point to 'Auto*' (was 'GTP2*' before)
#   - changed 'max_length' to config 'max_position_embeddings' (was 'n_positions' before)
#   - changed 'device' from 'cuda' to 'cpu'
#   - changed 'stride' to be 'max_length' (was 512 or 'max_length//2' before)
#   - removed 'with torch.no_grad()' and added global 'torch.set_grad_enabled(False)'
# Perhaps a simpler and cleaner way is given here:
#   https://huggingface.co/spaces/evaluate-metric/perplexity
#
# Usage:
#   1) run PPL of model from repo https://huggingface.co/HuggingFaceTB/SmolLM-135M
#        python3 ppl.py HuggingFaceTB/SmolLM-135M
#   2) run PPL of model from local directory './foo'
#        python3 ppl.py modified_repo
#   3) run with a seepdup of e.g. 4x
#        python3 ppl.py HuggingFaceTB/SmolLM-135M --speedup 4

import argparse
import transformer_tricks.tricks as tt
# to use ./tricks.py instead, uncomment the line below
# import tricks as tt

# define arguments
parser = argparse.ArgumentParser(description='calculate perplexity (PPL)')
parser.add_argument('repo', type=str, help='repo or local directory of LLM')
parser.add_argument('--speedup', type=int, default=1, help='specify the speedup')
parser.add_argument('--no_bars', action='store_true', help='turn off progress bars')

# get arguments
repo = parser.parse_args().repo
speedup = parser.parse_args().speedup
no_bars = parser.parse_args().no_bars

tt.perplexity(repo, speedup, no_bars)
