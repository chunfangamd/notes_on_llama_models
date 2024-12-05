#!/bin/bash

CHECKPOINT_DIR=/home/amd/chun/models/Llama3.2-11B-Vision-Instruct/
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun llama_models/scripts/multimodal_example_text_completion.py $CHECKPOINT_DIR