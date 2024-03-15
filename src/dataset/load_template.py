import os
import sys
import numpy as np
import ast

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_template(file_path):
    topology_path = os.path.join(file_path, 'topology.txt')
    vertices_path = os.path.join(file_path, 'vertices.txt')
    topology = ast.literal_eval(open(topology_path, 'r').read()) # topology is a list (face number,vertex number per face):(30,12)
    control_point_num = len(topology[0])/4 + 1

    # Process template data
    parameters = []
    vertex_idxs = np.zeros([len(open(vertices_path, 'r').readlines()), 3],dtype=np.int64)
    # vertices
    for i, l in enumerate(open(vertices_path, 'r')):
        value = l.strip().split(' ')
        _, a, b, c = value
        vertex_idxs[i] = [len(parameters), len(parameters)+1, len(parameters)+2]
        parameters.extend([float(a), float(b), float(c)])
    parameters = torch.tensor(parameters).squeeze()
    vertex_idxs = torch.from_numpy(vertex_idxs.astype(np.int64))

    # faces
    face_idxs = np.empty([len(topology), len(topology[0])])
    for i, patch in enumerate(topology):
        for j, k in enumerate(patch):
            face_idxs[i, j] = k
    face_idxs = torch.from_numpy(face_idxs.astype(np.int64))

    # curves
    n_curve = len(topology) * 4
    if len(topology[0]) == 12:
        curve_idxs = np.empty([n_curve, 4])
        for i, patch in enumerate(topology):
            curve_idxs[i*4, :] = patch[:4]
            curve_idxs[i*4+1, :] = patch[3:7]
            curve_idxs[i*4+2, :] = patch[6:10]
            curve_idxs[i*4+3, :] = [patch[9], patch[10], patch[11], patch[0]]
    elif len(topology[0]) == 28:
        curve_idxs = np.empty([n_curve, 8])
        for i, patch in enumerate(topology):
            curve_idxs[i*4, :] = patch[:8]
            curve_idxs[i*4+1, :] = patch[7:15]
            curve_idxs[i*4+2, :] = patch[14:22]
            curve_idxs[i*4+3, :] = [patch[21], patch[22], patch[23], patch[24], patch[25], patch[26], patch[27], patch[0]]
    curve_idxs = torch.from_numpy(curve_idxs.astype(np.int64))

    # symmetry
    xs, ys = [], []
    for line in open(os.path.join(file_path, 'symmetries.txt'), 'r'):
        x, y = line.strip().split(' ')
        xs.append(int(x))
        ys.append(int(y))
    symmetriy_idx = (xs, ys)
    symmetriy_idx = torch.tensor(symmetriy_idx)

    patch_kwargs = {
        "vertex_idx": vertex_idxs,
        "face_idx": face_idxs,
        "symmetriy_idx": symmetriy_idx,
    }
    curve_kwargs = {
        "vertex_idx": vertex_idxs,
        "curve_idx": curve_idxs,
    }

    return parameters, vertex_idxs, face_idxs, symmetriy_idx, curve_idxs

if __name__ == '__main__':
    # ` python src/dataset/load_template.py `
    file_path = "data/templates/sphere24"
    parameters, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(file_path)
    print(parameters.shape)