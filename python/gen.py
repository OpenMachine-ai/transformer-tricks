# run LLM from a HF repo (HuggingFace) or a local directory
#
# Usage:
#   1) run model from https://huggingface.co/HuggingFaceTB/SmolLM-135M
#        python3 gen.py HuggingFaceTB/SmolLM-135M
#   2) run model from local directory './foo'
#        python3 gen.py foo

import argparse
import transformer_tricks.tricks as tt
# to use ./tricks.py instead, uncomment the line below
# import tricks as tt

# define and get argument
parser = argparse.ArgumentParser(description='run LLM (aka generate tokens)')
parser.add_argument('repo', type=str, help='repo or local directory of LLM')
repo = parser.parse_args().repo

tt.hello_world(repo)
