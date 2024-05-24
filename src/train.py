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

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability

class Model(nn.Module):
    def __init__(self, template_params, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.register_buffer('template_params', template_params)

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_params)))

    def forward(self):
        vertices = self.template_params + self.displace

        return vertices.repeat(self.batch_size, 1)
        
def compute_loss(vertices, patches, curves, pcd_points, pcd_normals, pcd_area, mv_points, sample_num, symmetriy_idx, multi_view_weights, epoch_num):
    batch_size = patches.shape[0]
    face_num = patches.shape[1]
    st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(patches) # [b, patch_num, sample_num, 2]

    FA_rate = 0.0
    chamfer_weight_rate = 0.5

    # preprocessing
    # patches (B,face_num,cp_num,3)
    points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
    normals = coons_normals(st[..., 0], st[..., 1], patches)
    mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 
    pcd_points  = pcd_points.repeat(batch_size,1,1)
    mv_points = mv_points.repeat(batch_size,1,1)
    pcd_normals = pcd_normals.repeat(batch_size,1,1)
    pcd_area    = pcd_area.repeat(batch_size,1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss, normal_loss = area_weighted_chamfer_loss(
        mtds, points, normals, pcd_points, pcd_normals, chamfer_weight_rate, multi_view_weights
    )
    
    # curve chamfer loss
    # curves (b, curve_num, cp_num, 3)
    curves_s_num = 16
    linspace = torch.linspace(0, 1, curves_s_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(linspace, curves)
    curve_chamfer_loss, _, _, _ = curve_chamfer(curve_points, pcd_points)
    curve_chamfer_loss = curve_chamfer_loss.mean(1)

    # multi view curve loss
    _, _, mv_curve_loss, _ = curve_chamfer(curve_points, mv_points)
    mv_curve_loss = mv_curve_loss.mean(1)

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

    # curvature loss
    curvature_loss = curve_curvature_loss(curves, linspace)

    
    # loss = chamfer_loss + 0.01*overlap_loss + 2*planar_loss + 0.1*symmetry_loss
    stable_loss = chamfer_loss + 0.01*overlap_loss + 2.5*planar_loss + 0.1*symmetry_loss# + 0.012 * curvature_loss
    if epoch_num <= 50:
        loss = stable_loss + 0.012 * curvature_loss
                # + 0.1 * math.exp(-epoch_num/100) * normal_loss
                #+ 0.05 * mv_curve_loss 
    elif epoch_num > 50 and epoch_num <= 100:
        loss = stable_loss + 0.012 * curvature_loss + 0.1*normal_loss * math.exp((epoch_num-150)/100)
    else :
        loss = stable_loss* math.exp((100-epoch_num)/100) +  0.1*normal_loss * math.exp((epoch_num-150)/100) + 0.01*curvature_loss * math.exp((100-epoch_num)/100)
                #+ math.exp((epoch_num-450)/100) * curve_chamfer_loss
                #+ 0.05 * mv_curve_loss \

    return loss.mean()

def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)

def save_obj(filename, points):
    # points (p_num, 3)
    with open(filename, 'w') as file:
        # 遍历每个点，将其写入文件
        for point in points:
            # 格式化为OBJ文件中的顶点数据行
            file.write("v {} {} {}\n".format(*point))

def training(**kwargs):
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

    # load multi_view weights .pt
    mv_path = kwargs['mv_path']
    weight_path = os.path.join(mv_path, 'weights.pt')
    if os.path.exists(weight_path):
        mview_weights_ = torch.load(f'{model_path}/weights.pt')
        # 0~1 -> 1~2
        mview_weights = (mview_weights_/2 + 1).detach()
        mview_weights.requires_grad_(False)
    else:
        print(f"{weight_path} doesn't exist, mview_weights = None")
        mview_weights = None

    # multi-view points sample from pcd
    mv_points = multiview_sample(pcd_points, mview_weights_)

    # load template
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx =\
    template_params.to(device), vertex_idx.to(device), face_idx.to(device), symmetriy_idx.to(device), curve_idx.to(device)
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
        patches = vertices[:,face_idx] # (B, face_num, cp_num, 3)
        curves = vertices[:,curve_idx] # (B, curve_num, cp_num, 3)
        loss_params = {
            'vertices':vertices,
            'patches':patches,
            'curves':curves,
            'pcd_points':pcd_points,
            'pcd_normals':pcd_normals,
            'pcd_area':pcd_area,
            'mv_points':mv_points,
            'sample_num':sample_num,
            'symmetriy_idx':symmetriy_idx,
            'multi_view_weights':mview_weights,
            'epoch_num':i
        }
        loss = compute_loss(**loss_params)
        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0 or i == kwargs['epoch'] - 1:
            with torch.no_grad():
                output_path = kwargs['output_path']
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(f"{output_path}/training_save", exist_ok=True)
                write_curve_points(f"{output_path}/training_save/{i}_curve.obj", curves[0], control_point_num)
                write_obj(f"{output_path}/training_save/{i}_mesh.obj", patches[0], control_point_num)

                # curve probability
                # curves (b, curve_num, cp_num, 3)
                curve_sample_num = 16
                curves, curves_mask = curve_probability(mv_points, curves[0], curve_sample_num)
                torch.save(curves_mask, f'{output_path}/curves_mask.pt')
                # write_curve_points(f"{output_path}/training_save/{i}_curve_d.obj", curves, control_point_num)

                # save control points
                save_obj(f"{output_path}/control_points.obj", vertices[0])
        
if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="output_cat_test")
    parser.add_argument('--mv_path', type=str, default="render_utils/train_outputs")

    parser.add_argument('--epoch', type=int, default="151")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.02")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()
    training(**vars(args))