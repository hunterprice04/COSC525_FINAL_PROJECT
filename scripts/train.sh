#!/bin/bash
python3 submodules/transformers/examples/tensorflow/language-modeling/run_clm.py \
    --model_type small-transformer \
    --tokenizer_name gpt2 \
    --output_dir output \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
