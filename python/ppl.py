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

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch, sys, argparse

# define arguments
parser = argparse.ArgumentParser(description='calculate perplexity (PPL)')
parser.add_argument('repo', type=str, help='repo or local directory of LLM')
parser.add_argument('--speedup', type=int, default=1, help='specify the speedup')
parser.add_argument('--no_bars', action='store_true', help='turn off progress bars')

# get arguments
repo = parser.parse_args().repo
speedup = parser.parse_args().speedup
no_bars = parser.parse_args().no_bars

torch.set_grad_enabled(False)  # speed up torch

tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

# tokenize wikitext2
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = model.config.max_position_embeddings
stride = max_length  # before it was 512 or max_length // 2
seq_len = encodings.input_ids.size(1) // speedup

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride), disable=no_bars):
  end_loc = min(begin_loc + max_length, seq_len)
  trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
  input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cpu')
  target_ids = input_ids.clone()
  target_ids[:, :-trg_len] = -100
  outputs = model(input_ids, labels=target_ids)

  # loss is calculated using CrossEntropyLoss which averages over valid labels
  # N.B. the model only calculates loss over trg_len - 1 labels, because it
  # internally shifts the labels to the left by 1.
  neg_log_likelihood = outputs.loss
  nlls.append(neg_log_likelihood)

  prev_end_loc = end_loc
  if end_loc == seq_len:
    break

ppl = torch.exp(torch.stack(nlls).mean())
print('ppl:', ppl)
#print('nlls:', nlls)
