#!/bin/bash

model_name='rabbit2'
output_filename=outputs/${model_name}

python src/train_and_fit.py  \
--model_path=./data/models/${model_name} \
--output_path=./${output_filename} \
--prep_output_path=./${output_filename}/prep_outputs/train_outputs \
--epoch=201 \
# --template_path=./data/templates/donut
# --template_path=./data/templates/cup24

# python src/post_perceptual.py \
# --model_path=./data/models/${model_name} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/prep_outputs/train_outputs \
# --object_curve_num=20 \
# --template_path=./data/templates/donut

# python src/postprocess.py \
# --model_path=./data/models/${model_name} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/render_utils/train_outputs \
# --match_rate=0.6

# python eval/fitting_eval.py \
# --model_path=./data/models/${model_name} \
# --output_path=./${output_filename} \
# --th=0.02