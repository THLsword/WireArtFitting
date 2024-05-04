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

def post_processing(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # init param
    batch_size = kwargs["batch_size"]

    # load .npz point cloud
    model_path = kwargs['model_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_points = pcd_points.repeat(batch_size,1,1)

    # load template
    template_path = kwargs['template_path']
    _, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    vertex_idx, face_idx, curve_idx = vertex_idx.to(device), face_idx.to(device), curve_idx.to(device)
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))*10

    # load control points -> curves 
    output_path = kwargs['output_path']
    control_points = load_obj(f"{output_path}/control_points.obj").to(device)
    curves = control_points[curve_idx] # (curve_num, cp_num, 3)
    # delete useless curve or not
    if os.path.exists(f'{output_path}/curves_mask.pt') and kwargs['d_curve']:
        curves_mask = torch.load(f'{output_path}/curves_mask.pt')
        curves = curves[curves_mask]

    # find k points close to curves' sample points
    curve_num = curves.shape[0]
    curves = curves.repeat(batch_size, 1, 1, 1) # (b, curve_num, cp_num, 3)
    linspace = torch.linspace(0, 1, sample_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(linspace, curves)
    # _, pcd_idx, _, _ = curve_chamfer(curve_points, pcd_points)
    _, pdc_k_idx = curve_2_pcd_kchamfer(curve_points, pcd_points, kwargs['k']) # in losses.py
    pcd_idx = torch.unique(pdc_k_idx)
    sampled_pcd = pcd_points[0][pcd_idx]

    # each curves' points idx and cood
    review_idx = pdc_k_idx.view(curve_num, -1, kwargs['k']) # (1, 13440, 3) -> (96, 140, 3)
    curve_idx_list = []
    curve_cood_list = []
    for i in review_idx:
        curve_idx_list.append(torch.unique(i))
        cood = pcd_points[0][torch.unique(i)]
        curve_cood_list.append(cood)

    # load multi view points
    mv_path = kwargs['mv_path']
    mv_points = load_obj(f"{mv_path}/multi_view.obj").to(device)
    
    # match each curve's cood <-> mv points
    matched_points_list = []
    matched_points = torch.tensor([]).to(device)
    for i in curve_cood_list:
        matches = i.unsqueeze(1) == mv_points.unsqueeze(0)
        matches = matches.all(dim=2)
        matched_indices_tensor1 = matches.any(dim=1) # [curve_on_pcd num]
        matched_indices_tensor2 = matches.any(dim=0) # [mv num]
        matched_points_list.append(i[matched_indices_tensor1])

    # each curve points num after matching, then calculate rates
    curve_match_rate = []
    for i, points in enumerate(matched_points_list):
        rate = len(points) / len(curve_idx_list[i])
        curve_match_rate.append(rate)
    curve_match_rate = torch.tensor(curve_match_rate)
    curve_cood_list_thresh = [cood for i, cood in enumerate(curve_cood_list) if curve_match_rate[i] > kwargs['match_rate']]
    matched_points_list_thresh = [cood for i, cood in enumerate(matched_points_list) if curve_match_rate[i] > kwargs['match_rate']]
    curve_thresh_mask = curve_match_rate > kwargs['match_rate']
    
    # chamfer distance of curves' matched points <-> curves' sample points, calculate offset vecotr of each curve
    curve_matched_sample_points = curve_points[0][curve_thresh_mask] # (matched curve num, sample num, 3)
    mean_offsets = []
    for i, points in enumerate(matched_points_list_thresh):
        # i [n, 3], i <-> this curve sample points chamfer distance
        chamferloss_a, idx_a, _, _ = curve_chamfer(points.unsqueeze(0), curve_matched_sample_points[i].unsqueeze(0))
        idx_a = idx_a.squeeze()
        temp = points - curve_matched_sample_points[i][idx_a]
        mean_offsets.append(temp.mean(0))
    mean_offsets = torch.stack(mean_offsets)
    
    # pcd + offset vectors
    counter = 0
    for i, points in enumerate(curve_cood_list):
        if curve_thresh_mask[i]:
            points = points + mean_offsets[counter]
            matched_points = torch.cat([matched_points, points], dim=0)
            counter = counter + 1

    # curve + offset

    # save sampled_pcd
    save_obj(f"{output_path}/sampled_pcd.obj", sampled_pcd)
    save_obj(f"{output_path}/matched_points.obj", matched_points)

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="output_cat_test")
    parser.add_argument('--mv_path', type=str, default="render_utils/train_outputs")

    parser.add_argument('--epoch', type=int, default="1")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.01")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()
    post_processing(**vars(args)) # find curves in pcd