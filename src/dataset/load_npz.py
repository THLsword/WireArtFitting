import os
import sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_npz(file_path):
    npzfile = np.load(file_path)
    points = npzfile['points']
    normals = npzfile['normals']
    points = torch.tensor(points, dtype=torch.float32)
    normals = torch.tensor(normals, dtype=torch.float32)

    return points, normals


if __name__ == '__main__':
    # ` python src/dataset/load_npz.py `
    file_path = "data/models/cat.npz"
    points, normals = load_npz(file_path)
    print(points.shape) # (4096,3)
    print(normals.shape) # (4096,3)