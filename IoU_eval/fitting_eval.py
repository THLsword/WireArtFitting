import os
import tqdm
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image
from sklearn.decomposition import PCA
import open3d
from scipy.spatial.distance import cdist
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
from src.dataset.load_pcd import load_npz, load_obj
from src.dataset.load_template import load_template
from src.utils.patch_utils import *
from src.utils.losses import *
from src.utils.curve_utils import * 
from src.utils.mview_utils import multiview_sample, curve_probability

def main(**kwargs):
    th = kwargs['th']
    
    # load pcd
    model_path = kwargs['model_path']
    output_path = kwargs['output_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    # pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_maen = abs(pcd_points).mean(0) # [3]
    average_distance = torch.norm(pcd_points, dim=1).mean()

    # load tamplate
    batch_size = 1
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)

    # load mesh
    output_path = kwargs['output_path']
    control_points = load_obj(f"{output_path}/control_points.obj")
    patches = control_points[face_idx]
    face_num = patches.shape[0]
    st = torch.empty(face_num, 13**2, 2).uniform_().to(patches)
    points  = coons_points(st[..., 0], st[..., 1], patches)
    points = points.view(-1,3)

    # compute chamfer distance d1 d2
    dist_matrix = cdist(np.array(pcd_points), np.array(points), metric='euclidean')
    d1 = np.min(dist_matrix, axis=1)
    d2 = np.min(dist_matrix, axis=0)

    print(d1.shape)
    print(d2.shape)

    recall = float(sum(d < th for d in d2)) / float(len(d2))
    precision = float(sum(d < th for d in d1)) / float(len(d1))
    if recall+precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0

    print(fscore)


if __name__ == '__main__':
    model_name = 'rabbit_noise2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--th', type=float, default=0.02)

    parser.add_argument('--model_path', type=str, default=f"data/models/{model_name}")
    parser.add_argument('--output_path', type=str, default=f"outputs/{model_name}")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")

    args = parser.parse_args()

    main(**vars(args))