#!/bin/bash

model_name='cat'
output_filename=outputs/${model_name}

python src/preprocessing.py \
--DATA_DIR=./data/models/${model_name} \
--RENDER_SAVE_DIR=./${output_filename}/render_utils/render_outputs \
--ALPHA_SAVE_DIR=./${output_filename}/render_utils/alpha_outputs \
--TRAIN_SAVE_DIR=./${output_filename}/render_utils/train_outputs \
--FILENAME=model_normalized_4096.npz \
--ALPHA_SIZE=45.0 \
--EXPAND_SIZE=1 

# python src/train_and_fit.py       \
# --model_path=./data/models/${model_name} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/render_utils/train_outputs \
# --epoch=201

# python src/postprocess.py \
# --model_path=./data/models/${model_name} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/render_utils/train_outputs \
# --match_rate=0.6