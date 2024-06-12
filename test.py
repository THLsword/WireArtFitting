import os
import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image

pc = torch.randint(10, (5,3), dtype=float) # (5,3)
# 經過knn msak——離pointcloud較遠的那些點
normals = torch.randint(10, (7,3), dtype=float) # (7,3) normal是deform mesh的
masked_mid_points = torch.randint(10, (7,3), dtype=float) # (7,3)
thres = 0.9
print("normals ", normals.shape)
print("masked_mid_points ", masked_mid_points.shape)

# 計算deform mesh上所有點到pcd所有點的向量和距離 pc->mesh
displacement = masked_mid_points[:, None, :] - pc[:, :3]
distance = displacement.norm(dim=-1)
print("displacement ", displacement.shape) # [7, 5, 3]
print("distance ", distance.shape) # [7,5]
# 單位向量*法向量，然後求和
mask = (torch.abs(torch.sum((displacement / distance[:, :, None]) * normals[:, None, :], dim=-1)) > thres) # (7,5)
print("mask ", mask)

dmin, argmin = distance.min(dim=-1)
print("dmin: ", dmin.shape)
print("argmin: ", argmin.shape)

pc_per_face_masked = pc[argmin, :].clone()
# pc_per_face = torch.zeros(mid_points.shape[0], 6, dtype = float)
print(pc_per_face_masked)
print(pc_per_face_masked.shape)

non_inf_mask = ~torch.isinf(dmin)
print(non_inf_mask.shape)