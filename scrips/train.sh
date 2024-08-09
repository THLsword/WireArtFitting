#!/bin/bash

animel='cat'
output_filename=outputs/${animel}_6v

python src/train_copy.py       \
--model_path=./data/models/${animel} \
--output_path=./${output_filename} \
--mv_path=./${output_filename}/render_utils/train_outputs \
--epoch=201

# python src/postprocess.py \
# --model_path=./data/models/${animel} \
# --output_path=./${output_filename} \
# --mv_path=./${output_filename}/render_utils/train_outputs \
# --match_rate=0.6