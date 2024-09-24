# convert an LLM to use flashNorm, see paper https://arxiv.org/abs/2407.09577
#
# Usage:
#   1) flashify https://huggingface.co/HuggingFaceTB/SmolLM-135M
#        python3 flashify.py HuggingFaceTB/SmolLM-135M
#        # this will create a local dir called 'SmolLM-135M_flashNorm'
#   2) same as (1) but name the local dir 'foo'
#        python3 flashify.py HuggingFaceTB/SmolLM-135M --out foo

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, argparse, os
import transformer_tricks.tricks as tt
# to use ./tricks.py instead, uncomment the line below
# import tricks as tt

torch.set_grad_enabled(False)  # speed up PyTorch

# define arguments
parser = argparse.ArgumentParser(description='convert LLM to use flashNorm')
parser.add_argument('repo', type=str, help='repo or local directory of original LLM')
parser.add_argument('-o', '--out', type=str, help='output directory')

# get arguments
repo = parser.parse_args().repo
out_dir = parser.parse_args().out
if out_dir == None:  # append '_flashNorm' if no output dir is defined
  out_dir = os.path.basename(repo) + '_flashNorm'

# load model and flashify it
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)
model = tt.flashify(model)

# save model and tokenizer in local directory 'out_dir'
tokenizer.save_pretrained(out_dir, from_pt=True)
model.save_pretrained(out_dir, from_pt=True)
