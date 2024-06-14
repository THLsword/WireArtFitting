#!/bin/bash

animel=cat
output_filename=output_${animel}_6v

# python render_utils/single_model_rendering.py \
# --DATA_DIR=./data/models/${animel} \
# --SAVE_DIR=./${output_filename}/render_utils/render_outputs

# python render_utils/dataset_alphashape.py \
# --DATA_DIR=./${output_filename}/render_utils/render_outputs \
# --SAVE_DIR=./${output_filename}/render_utils/alpha_outputs

# python render_utils/alphashape_expand.py \
# --expend_size=2 \
# --render_DIR=./${output_filename}/render_utils/render_outputs \
# --alphashape_DIR=./${output_filename}/render_utils/alpha_outputs \
# --SAVE_DIR=./${output_filename}/render_utils/expand_outputs

# python render_utils/demo_deform.py \
# --DATA_DIR=./data/models/${animel} \
# --GT_DIR=./${output_filename}/render_utils/expand_outputs \
# --SAVE_DIR=./${output_filename}/render_utils/train_outputs 

# python src/train.py       \
# --model_path=./data/models/${animel} \
# --output_path=./${output_filename} \
# --mv_path=./${output_filename}/render_utils/train_outputs \
# --epoch=201

python src/postprocess.py \
--model_path=./data/models/${animel} \
--output_path=./${output_filename} \
--mv_path=./${output_filename}/render_utils/train_outputs