#!/bin/bash

# flashify LLMs and run inference and perplexity to make sure that
# the flashified models are equivalent to the original ones
#
# Usage:
#   ./run_all.sh

# convert models to flashNorm
python3 flashify.py HuggingFaceTB/SmolLM-135M
python3 flashify.py HuggingFaceTB/SmolLM-360M
#python3 flashify.py HuggingFaceTB/SmolLM-1.7B
#python3 flashify.py microsoft/Phi-3-mini-4k-instruct

# run models
python3 gen.py HuggingFaceTB/SmolLM-135M
python3 gen.py               SmolLM-135M_flashNorm
python3 gen.py HuggingFaceTB/SmolLM-360M
python3 gen.py               SmolLM-360M_flashNorm
#python3 gen.py HuggingFaceTB/SmolLM-1.7B
#python3 gen.py               SmolLM-1.7B_flashNorm
#python3 gen.py microsoft/Phi-3-mini-4k-instruct
#python3 gen.py           Phi-3-mini-4k-instruct_flashNorm

# measure perplexity
python3 ppl.py HuggingFaceTB/SmolLM-135M           --speedup 16 --no_bars
python3 ppl.py               SmolLM-135M_flashNorm --speedup 16 --no_bars
python3 ppl.py HuggingFaceTB/SmolLM-360M           --speedup 16 --no_bars
python3 ppl.py               SmolLM-360M_flashNorm --speedup 16 --no_bars
#python3 ppl.py HuggingFaceTB/SmolLM-1.7B           --speedup 64 --no_bars
#python3 ppl.py               SmolLM-1.7B_flashNorm --speedup 64 --no_bars
#python3 ppl.py microsoft/Phi-3-mini-4k-instruct           --speedup 64
#python3 ppl.py           Phi-3-mini-4k-instruct_flashNorm --speedup 64

# clean up
rm -Rf SmolLM*_flashNorm

# TODO: add more LLMs
#python3 gen.py stabilityai/stablelm-2-1_6b  # doesn't use RMSNorm, but LayerNorm
#python3 gen.py meta-llama/Meta-Llama-3.1-8B
#python3 gen.py mistralai/Mistral-7B-v0.3

# Notes for running larger models:
#   - To run llama and other semi-secret LLMs, you first have to type the following:
#       huggingface-cli login
#     above will ask you for the hf_token, which is the same you use e.g. in colab
#
#   - On MacBook, open the 'Activity Monitor' and check your memory usage. If your
#     MacBook has only 8GB of DRAM, then you have only about 6GB available. Many LLMs
#     use float32, so a 1.5B model needs at least 6GB of DRAM.
#
#   - Running gen.py is limited by DRAM bandwidth, not compute. Running ppl.py is
#     usually limited by compute (rather than by memory bandwidth), so only having
#     8GB of DRAM is likely not an issue for running ppl.py on larger LLMs. That's
#     because ppl.py doesn't do the auto-regressive generation phase but only the
#     prompt phase (where all input tokens are batched).
#
#   - The models get cached on your system at ~/.cache/huggingface, which can grow
#     very big, see du -h -d 3 ~/.cache/huggingface
