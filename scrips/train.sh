#!/bin/bash

(
trap 'kill 0' SIGINT;

tensorboard --logdir apes_log --port 1122 --bind_all & \

python src/train.py \
--filename cube_data_batch8_02
# --filename modelnet_local_02

# python src/train.py --conf-path configs/animal.conf
)
