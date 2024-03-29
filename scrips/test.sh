#!/bin/bash
save_dir='apes_log'

# file_path="modelnet_local_02"
file_path="cube_data_batch8_02"

# ckpt_name="last.ckpt"
ckpt_name="last.ckpt"

python src/test.py \
--checkpoint_path ${save_dir}/${file_path}/checkpoints/${ckpt_name} 
