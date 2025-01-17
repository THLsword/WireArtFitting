#!/bin/bash

animel='dog2'
output_filename=outputs/${animel}_6v

python src/train_and_fit.py       \
--model_path=./data/models/${animel} \
--output_path=./${output_filename} \
--prep_output_path=./${output_filename}/render_utils/train_outputs \
--epoch=201 \
--template_path=./data/templates/cube24

# python src/postprocess.py \
# --model_path=./data/models/${animel} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/render_utils/train_outputs \
# --match_rate=0.6

# python src/post_perceptual.py \
# --model_path=./data/models/${animel} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/render_utils/train_outputs \
# --object_curve_num=35 \
# # --template_path=./data/templates/cube24

# python eval/fitting_eval.py \
# --model_path=./data/models/${animel} \
# --output_path=./${output_filename} \
# --th=0.02