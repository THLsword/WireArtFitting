#!/bin/bash

python render_utils/single_model_rendering.py \
--DATA_DIR "./data/models/cat"
python render_utils/dataset_alphashape.py
python render_utils/alphashape_expand.py --expend_size=2
python render_utils/demo_deform.py --DATA_DIR="./data/models/cat"