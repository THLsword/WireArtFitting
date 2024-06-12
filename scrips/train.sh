#!/bin/bash

animel='rabbit'

# python src/train.py       \
# --model_path=./data/models/${animel} \
# --output_path=./output_${animel} \
# --mv_path=./output_${animel}/render_utils/train_outputs \
# --epoch=201

python src/postprocess.py \
--model_path=./data/models/${animel} \
--output_path=./output_${animel} \
--mv_path=./output_${animel}/render_utils/train_outputs \
--match_rate=0.6