import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.utils.postprocess_utils import render
from PIL import Image
import numpy as np
import alphashape
import threading
import psutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import networkx as nx


# x1 [b,N,3]
# x2 [b,N,3]
x1_norm = x1.pow(2).sum(-1, keepdim=True) # [8, 4374, 1]
x2_norm = x2.pow(2).sum(-1, keepdim=True) # [8, 4096, 1]

res = torch.baddbmm(
    x2_norm.transpose(-2, -1),
    x1,
    x2.transpose(-2, -1),
    alpha=-2
).add_(x1_norm).clamp_min_(1e-10).sqrt_()
# ||(x1 - x2)|| = ((x1 ^ 2) - 2 * (x1 * x2) + (x2 ^ 2)) ^ (1/2)
# [b, n_sample_points, n_mesh_points]