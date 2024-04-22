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
        
def compute_loss(vertices, patches, curves, pcd_points, pcd_normals, pcd_area, pcd_s, sample_num, symmetriy_idx, multi_view_weights, epoch_num):
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
    pcd_s = pcd_s.repeat(batch_size,1,1)
    pcd_normals = pcd_normals.repeat(batch_size,1,1)
    pcd_area    = pcd_area.repeat(batch_size,1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss, normal_loss = area_weighted_chamfer_loss(
        mtds, points, normals, pcd_points, pcd_normals, chamfer_weight_rate, multi_view_weights
    )
    
    # curve chamfer loss
    # curves (b, curve_num, cp_num, 3)
    curves_s_num = 16
    curves_s = torch.linspace(0, 1, curves_s_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(curves_s, curves)
    curve_chamfer_loss, _, _, _ = curve_chamfer(curve_points, pcd_points)
    curve_chamfer_loss = curve_chamfer_loss.mean(1)

    # multi view curve loss
    _, _, mv_curve_loss, _ = curve_chamfer(curve_points, pcd_s)
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

    # loss = chamfer_loss + 0.01*overlap_loss + 2*planar_loss + 0.1*symmetry_loss
    if epoch_num <= 200:
        loss = chamfer_loss + 0.01*overlap_loss + 2.0*planar_loss + 0.1*symmetry_loss \
                # + 0.1 * math.exp(-epoch_num/100) * normal_loss
                #+ 0.05 * mv_curve_loss 
    else:
        loss = chamfer_loss + 0.01*overlap_loss + 2.0*planar_loss + 0.1*symmetry_loss + \
               0.1*math.exp((epoch_num-400)/100) * normal_loss 
            #    + math.exp((epoch_num-450)/100) * curve_chamfer_loss
               #+ 0.05 * mv_curve_loss \

    if epoch_num > 400:
        loss = math.exp((epoch_num-500)/100) * curve_chamfer_loss + 0.1*symmetry_loss

    return loss.mean()

def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)

def save_obj(filename, cps):
    # cps (p_num, 3)
    with open(filename, 'w') as file:
        # 遍历每个点，将其写入文件
        for point in cps:
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

    # load weights .pt
    mview_weights_ = torch.load(f'{model_path}/weights.pt')
    mview_weights = (mview_weights_/2 + 1).detach()
    mview_weights.requires_grad_(False)

    # multi-view sample pcd
    pcd_s = multiview_sample(pcd_points, mview_weights_)

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
        patches = vertices[:,face_idx] # (B, face_num, cp_num, 3)
        curves = vertices[:,curve_idx] # (B, curve_num, cp_num, 3)
        loss_params = {
            'vertices':vertices,
            'patches':patches,
            'curves':curves,
            'pcd_points':pcd_points,
            'pcd_normals':pcd_normals,
            'pcd_area':pcd_area,
            'pcd_s':pcd_s,
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

        if i % 100 == 0 or i == kwargs['epoch'] - 1:
            with torch.no_grad():
                output_path = kwargs['output_path']
                os.makedirs(output_path, exist_ok=True)
                write_curve_points(f"{output_path}/{i}_curve.obj", curves[0], control_point_num)
                write_obj(f"{output_path}/{i}_mesh.obj", patches[0], control_point_num) 

                # curve probability
                # curves (b, curve_num, cp_num, 3)
                curve_s_num = 16
                topk_num = 80
                curves, rate_mask = curve_probability(pcd_s, curves[0], curve_s_num, topk_num)
                torch.save(rate_mask, f'{output_path}/rate_mask.pt')
                write_curve_points(f"{output_path}/{i}_curve_d.obj", curves, control_point_num)

                # save control points
                save_obj(f"{output_path}/control_points.obj", vertices[0])
        
def post_processing(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # init
    batch_size = kwargs["batch_size"]

    # load .npz point cloud
    model_path = kwargs['model_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_points = pcd_points.repeat(batch_size,1,1)

    # load template
    template_path = kwargs['template_path']
    _, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    vertex_idx = vertex_idx.to(device)
    face_idx = face_idx.to(device)
    curve_idx = curve_idx.to(device)
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))*10

    # load control points
    output_path = kwargs['output_path']
    cps = load_obj(f"{output_path}/control_points.obj").to(device)
    curves = cps[curve_idx] # (curve_num, cp_num, 3)
    if os.path.exists(f'{output_path}/rate_mask.pt') and kwargs['d_curve']:
        rate_mask = torch.load(f'{output_path}/rate_mask.pt')
        curves = curves[rate_mask]

    # find curves on pcd
    curve_num = curves.shape[0]
    curves = curves.repeat(batch_size, 1, 1, 1) # (b, curve_num, cp_num, 3)
    curves_s = torch.linspace(0, 1, sample_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(curves_s, curves)
    _, pcd_idx, _, _ = curve_chamfer(curve_points, pcd_points)
    _, pdc_k_idx = curve_2_pcd(curve_points, pcd_points, kwargs['k']) # in losses.py
    pcd_idx = torch.unique(pdc_k_idx)
    sampled_pcd = pcd_points[0][pcd_idx]

    # each curve points num
    review_idx = pdc_k_idx.view(curve_num, -1, kwargs['k'])
    curve_p_idx = []
    curve_p_cood = []
    for i in review_idx:
        curve_p_idx.append(torch.unique(i))
        cood = pcd_points[0][torch.unique(i)]
        curve_p_cood.append(cood)

    # load multi view points
    mv_path = kwargs['mv_path']
    mv_points = load_obj(f"{mv_path}/multi_view.obj").to(device)
    
    # match
    matched_points_list = []
    matched_points = torch.tensor([]).to(device)
    for i in curve_p_cood:
        matches = i.unsqueeze(1) == mv_points.unsqueeze(0)
        matches = matches.all(dim=2)
        matched_indices_tensor1 = matches.any(dim=1) # [curve_on_pcd num]
        matched_indices_tensor2 = matches.any(dim=0) # [mv num]
        matched_points_list.append(i[matched_indices_tensor1])

    # each curve points num after match
    curve_match_rate = []
    for i, points in enumerate(matched_points_list):
        rate = len(points) / len(curve_p_idx[i])
        curve_match_rate.append(rate)
    curve_match_rate = torch.tensor(curve_match_rate)
    curve_match_cood = [cood for i, cood in enumerate(curve_p_cood) if curve_match_rate[i] > kwargs['match_rate']]
    curve_match_idx = curve_match_rate > kwargs['match_rate']
    
    # match again
    curve_matched_p = []
    for i in curve_match_cood:
        matches = i.unsqueeze(1) == mv_points.unsqueeze(0)
        matches = matches.all(dim=2)
        matched_indices_tensor1 = matches.any(dim=1) # [curve_on_pcd num]
        matched_indices_tensor2 = matches.any(dim=0) # [mv num]
        # matched_points = torch.cat([matched_points, i[matched_indices_tensor1]], dim=0)
        curve_matched_p.append(i[matched_indices_tensor1])
    
    # curves' matched points <-> curves' sample points
    curve_matched_sample_points = curve_points[0][curve_match_idx] # (matched curve num, sample num, 3)
    mean_offsets = []
    for i, points in enumerate(curve_matched_p):
        # i [n, 3], i <-> this curve sample points chamfer distance
        chamferloss_a, idx_a, _, _ = curve_chamfer(points.unsqueeze(0), curve_matched_sample_points[i].unsqueeze(0))
        idx_a = idx_a.squeeze()
        temp = points - curve_matched_sample_points[i][idx_a]
        mean_offsets.append(temp.mean(0))
    mean_offsets = torch.stack(mean_offsets)
    # print(mean_offsets)
    
    # pcd + offset vectors
    print(len(curve_p_cood))
    print(curve_matched_sample_points.shape)
    print(mean_offsets.shape)
    counter = 0
    for i, points in enumerate(curve_p_cood):
        if curve_match_idx[i]:
            points = points + mean_offsets[counter]
            matched_points = torch.cat([matched_points, points], dim=0)
            counter = counter + 1

        
    # save sampled_pcd
    save_obj(f"{output_path}/sampled_pcd.obj", sampled_pcd)
    save_obj(f"{output_path}/matched_points.obj", matched_points)

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    # parser.add_argument('--template_path', type=str, default="data/templates/animel122")
    parser.add_argument('--output_path', type=str, default="output_cat_24_m")
    parser.add_argument('--mv_path', type=str, default="render_utils/train_outputs")

    parser.add_argument('--epoch', type=int, default="401")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.01")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()
    # training(**vars(args))
    post_processing(**vars(args)) # find curves in pcd