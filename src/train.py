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

from dataset.load_npz import load_npz
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import *

class Model(nn.Module):
    def __init__(self, template_params, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.register_buffer('template_params', template_params)

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_params)))

    def forward(self):
        vertices = self.template_params + self.displace

        return vertices.repeat(self.batch_size, 1)
        
def compute_loss(vertices, patches, pcd_points, pcd_normals, pcd_area, sample_num, symmetriy_idx, epoch_num):
    st = torch.empty(
        patches.shape[0],
        patches.shape[1],
        sample_num**2,
        2
    ).uniform_().to(patches) # [b, patch_num, sample_num, 2]
    FA_rate = 0.0
    chamfer_weight_rate = 0.5

    # preprocessing
    points = coons_points(st[..., 0], st[..., 1], patches) # [b, patch, cp, 3]
    normals = coons_normals(st[..., 0], st[..., 1], patches)
    mtds = coons_mtds(st[..., 0], st[..., 1], patches)
    pcd_points = pcd_points.repeat(points.shape[0],1,1)
    pcd_normals = pcd_normals.repeat(points.shape[0],1,1)
    pcd_area = pcd_area.repeat(points.shape[0],1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss, normal_loss = area_weighted_chamfer_loss(
        mtds, points, normals, pcd_points, pcd_normals, chamfer_weight_rate
    )

    # flatness loss
    planar_loss = planar_patch_loss(st, points, mtds)

    # Conciseness loss
    overlap_loss = patch_overlap_loss(mtds, pcd_area)

    # Orthogonality loss
    perpendicular_loss = curve_perpendicular_loss(patches)

    # flatness loss
    FA_loss = flatness_area_loss(st, points, mtds)

    # symmetry loss
    symmetry_loss = torch.zeros(1).to(patches)
    symmetry_loss = patch_symmetry_loss(symmetriy_idx[0], symmetriy_idx[1], vertices)

    # loss = chamfer_loss + 0.01*overlap_loss + 2*planar_loss + 0.1*symmetry_loss
    if epoch_num <= 200:
        loss = chamfer_loss + 0.0*overlap_loss + 0*planar_loss + 0.1*symmetry_loss + 0.1*normal_loss * math.exp(-epoch_num/100)
    else:
        loss = chamfer_loss + 0.0*overlap_loss + 0*planar_loss + 0.1*symmetry_loss + 0.1*normal_loss * math.exp((epoch_num-400)/100)

    return loss.mean()

def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)


def main(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # load .npz point cloud
    model_path = kwargs['model_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)

    # load template
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    template_params = template_params.to(device)
    vertex_idx = vertex_idx.to(device)
    face_idx = face_idx.to(device)
    symmetriy_idx = symmetriy_idx.to(device)
    curve_idx = curve_idx.to(device)
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))

    # train
    batch_size = kwargs['batch_size']
    control_point_num = 4
    model = Model(template_params, batch_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'], betas=(0.5, 0.99))
    loop = tqdm.tqdm(list(range(0, kwargs['epoch'])))

    for i in loop:
        vertices = model()
        vertices = vertices.view(batch_size,-1,3)
        patches = vertices[:,face_idx]
        curves = vertices[:,curve_idx]
        loss = compute_loss(vertices, patches, pcd_points, pcd_normals, pcd_area, sample_num, symmetriy_idx, i)
        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with torch.no_grad():
                output_path = kwargs['output_path']
                os.makedirs(output_path, exist_ok=True)
                write_curve_points(f"{output_path}/{i}_curve.obj", curves[0], control_point_num)
                write_obj(f"{output_path}/{i}_mesh.obj", patches[0], control_point_num) 


if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    # parser.add_argument('--template_path', type=str, default="data/templates/animel122")
    parser.add_argument('--output_path', type=str, default="output_cat_24")

    parser.add_argument('--epoch', type=int, default="401")
    parser.add_argument('--batch_size', type=int, default="1")
    parser.add_argument('--learning_rate', type=float, default="0.02")

    args = parser.parse_args()
    main(**vars(args))