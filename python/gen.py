# run LLM from a HF repo (HuggingFace) or a local directory
#
# Usage:
#   1) run model from https://huggingface.co/HuggingFaceTB/SmolLM-135M
#        python3 gen.py HuggingFaceTB/SmolLM-135M
#   2) run model from local directory './foo'
#        python3 gen.py foo

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, argparse

# define and get argument
parser = argparse.ArgumentParser(description='run LLM (aka generate tokens)')
parser.add_argument('repo', type=str, help='repo or local directory of LLM')
repo = parser.parse_args().repo

tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)
# to use FP16 or bfloaf: torch_dtype=torch.float16, torch_dtype=torch.bfloat
# note: FP16 is 30x slower than FP32 on my Mac M1, not sure why

prompt = 'Once upon a time there was'
inp = tokenizer.encode(prompt, return_tensors='pt').to('cpu')
out = model.generate(inp, pad_token_id=0, max_new_tokens=4).ravel()
print(tokenizer.decode(out))

# TODO: especially for Phi-3, set verbosity to quiet as follows
#  transformers.logging.set_verbosity_error()
