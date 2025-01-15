import os
import sys
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_npz(file_path):
    npz_path = ''
    if os.path.exists(os.path.join(file_path, "model_normalized_4096.npz")):
        npz_path = os.path.join(file_path, "model_normalized_4096.npz")
    elif os.path.exists(os.path.join(file_path, "model_normalized_5000.npz")):
        npz_path = os.path.join(file_path, "model_normalized_5000.npz")
    else:
        raise FileNotFoundError(f"model.npz文件不存在。")
    npzfile = np.load(npz_path)
    points = npzfile['points']
    normals = npzfile['normals']
    points = torch.tensor(points, dtype=torch.float32)
    normals = torch.tensor(normals, dtype=torch.float32)

    area_path = os.path.join(file_path, "model_normalized_area.json")
    all_areas = []
    with open(area_path, 'r') as f:
        area = json.load(f)['area']
        all_areas.append(area)
    all_areas = torch.tensor(all_areas, dtype=torch.float32)

    return points, normals, all_areas

def load_obj(file_path):
    # 定义一个空列表来存储顶点坐标
    vertices = []

    # 打开.obj文件进行读取
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
    return torch.tensor(vertices)


if __name__ == '__main__':
    # ` python src/dataset/load_npz.py `
    file_path = "data/models/cat"
    points, normals, all_areas = load_npz(file_path)
    print(points.shape) # (4096,3)
    print(normals.shape) # (4096,3)
    print(all_areas)