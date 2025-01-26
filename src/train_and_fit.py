import os
from tqdm import tqdm
import math
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

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability
from utils.save_data import save_img, save_obj, save_lr_fig, save_loss_fig

from model.model_interface import Model

def lr_lambda(epoch):
    warm_epoch = 50
    k=5
    if epoch < warm_epoch:
        # return math.exp((epoch - warm_epoch) / k)
        return 0.1
    else:
        # return 0.99 ** (epoch - warm_epoch)
        return 1.0


def compute_loss(cp_coord, patches, curves, pcd_points, pcd_normals, pcd_area, prep_points, sample_num, tpl_sym_idx, prerp_weights_scaled, i: int, epoch_num: int):
    batch_size = patches.shape[0]
    face_num = patches.shape[1]
    st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(patches) # [b, patch_num, sample_num, 2]

    chamfer_weight_rate = 0.5

    # preprocessing
    # patches (B,face_num,cp_num,3)
    points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
    normals = coons_normals(st[..., 0], st[..., 1], patches)
    mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 
    pcd_points  = pcd_points.repeat(batch_size,1,1)
    prep_points = prep_points.repeat(batch_size,1,1)
    pcd_normals = pcd_normals.repeat(batch_size,1,1)
    pcd_area    = pcd_area.repeat(batch_size,1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss, normal_loss = area_weighted_chamfer_loss(
        mtds, points, normals, pcd_points, pcd_normals, chamfer_weight_rate, prerp_weights_scaled
    )
    
    # normal loss weight
    if i <= 50: 
        normal_loss = 0.1 * normal_loss * math.exp((i-50)/100)
    else:
        normal_loss = 0.1 * normal_loss

    # curve chamfer loss
    # curves (b, curve_num, cp_num, 3)
    curves_s_num = 16
    linspace = torch.linspace(0, 1, curves_s_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(linspace, curves)
    curve_chamfer_loss, _, _, _ = curve_chamfer(curve_points, pcd_points*1.01)
    curve_chamfer_loss = curve_chamfer_loss.mean(1)

    # multi view curve loss
    _, _, mv_curve_loss, _ = curve_chamfer(curve_points, prep_points)
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
    symmetry_loss = patch_symmetry_loss(tpl_sym_idx[0], tpl_sym_idx[1], cp_coord)

    # curvature loss
    curvature_loss = curve_curvature_loss(curves, linspace)

    # beam gap loss
    thres = 0.9
    beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, thres)
    
    # # loss = chamfer_loss + 0.01*overlap_loss + 2*planar_loss + 0.1*symmetry_loss
    # stable_loss = chamfer_loss + 2.5*planar_loss + 0.1*symmetry_loss# + 0.012 * curvature_loss
    # if i <= 50:
    #     loss = stable_loss + 0.2 * beam_gap_loss# + 0.012 * curvature_loss
    #             # + 0.1 * math.exp(-i/100) * normal_loss
    #             #+ 0.05 * mv_curve_loss 
    # elif i > 50 and i <= 100:
    #     loss = stable_loss + 0.012 * curvature_loss + 0.1*normal_loss * math.exp((i-150)/100) + 0.2 * beam_gap_loss
    # else :
    #     loss = stable_loss* math.exp((100-i)/100) +  0.1*normal_loss * math.exp((i-150)/100) + 0.2 * beam_gap_loss + 0.01*curvature_loss * math.exp((100-i)/100)
    #             #+ math.exp((i-450)/100) * curve_chamfer_loss
    #             #+ 0.05 * mv_curve_loss \

    # test loss
    # stable_loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss + normal_loss + 0.2 * beam_gap_loss
    stable_loss =  chamfer_loss + 2.0 *planar_loss + 0.1*symmetry_loss + 0.7 *normal_loss + 0.30 * beam_gap_loss# + 0.002 * curvature_loss# + 0.005*overlap_loss
    loss = stable_loss

    return loss.mean()

def training(**kwargs):
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # create path folder
    output_path = kwargs['output_path']
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/training_save", exist_ok=True)
    
    # load .npz point cloud
    model_path = kwargs['model_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_maen = abs(pcd_points).mean(0) # [3]

    # load preprocessing output: 'weights.pt'
    prep_output_path = kwargs['prep_output_path']
    weight_path = os.path.join(prep_output_path, 'weights.pt')
    if os.path.exists(weight_path):
        prep_weights = torch.load(weight_path)
        # 0~1 -> -0.25~0.25
        prerp_weights_scaled = ((prep_weights - 0.5)/2).detach()
        prerp_weights_scaled.requires_grad_(False)
    else:
        print(f"{weight_path} doesn't exist, prerp_weights_scaled = None")
        prerp_weights_scaled = None

    # multi-view points sample from pcd
    prep_points = multiview_sample(pcd_points, prep_weights) # (M, 3)

    # load template
    template_path = kwargs['template_path']
    tpl_params, tpl_v_idx, tpl_f_idx, tpl_sym_idx, tpl_c_idx = load_template(template_path)
    tpl_params, tpl_v_idx, tpl_f_idx, tpl_sym_idx, tpl_c_idx =\
    tpl_params.to(device), tpl_v_idx.to(device), tpl_f_idx.to(device), tpl_sym_idx.to(device), tpl_c_idx.to(device)
        # Number of samples on one face of the template
    sample_num = int(np.ceil(np.sqrt(pcd_points.shape[0] / tpl_f_idx.shape[0])))
        # Scale the template to a size similar to that of the point cloud
    template_mean = abs(tpl_params.view(-1,3)).mean(0) # [3]
    tpl_params = (tpl_params.view(-1,3) / template_mean * pcd_maen)

    # train
    BATCH_SIZE = kwargs['batch_size']
    CTRL_P_NUM = 4 # number of control points on one curve
    model = Model(tpl_params.repeat(BATCH_SIZE, 1, 1)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'], betas=(0.9, 0.99))
    loop = tqdm(range(kwargs['epoch']))
    scheduler = LambdaLR(optimizer, lr_lambda)

    loss_list = []
    lr_list = []
    for i in loop:
        model.train()
        indices = torch.randperm(pcd_points.size(0))  # 生成一个从 0 到 1023 的随机排列索引
        shuffled_pcd_points = pcd_points[indices]

        cp_coord = model(shuffled_pcd_points.repeat(BATCH_SIZE, 1, 1), prep_points.repeat(BATCH_SIZE, 1, 1)) # MLP:(B, N), review to (B, -1, 3)
        patches = cp_coord[:,tpl_f_idx] # (B, face_num, cp_num, 3)
        curves = cp_coord[:,tpl_c_idx] # (B, curve_num, cp_num, 3)
        prerp_weights_scaled = 1 + (prerp_weights_scaled * i / kwargs['epoch']) # 0.75~1.25
        loss = compute_loss(
            cp_coord=cp_coord, # [B, n, 3]
            patches=patches,
            curves=curves,
            pcd_points=pcd_points,
            pcd_normals=pcd_normals,
            pcd_area=pcd_area,
            prep_points=prep_points,
            sample_num=sample_num,
            tpl_sym_idx=tpl_sym_idx,
            prerp_weights_scaled=prerp_weights_scaled.to(device),
            i=i,
            epoch_num=kwargs['epoch']
        )
        loop.set_description('Loss: %.4f' % (loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print(f"Epoch {i + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # record loss and learning rate
        with torch.no_grad():
            loss_list.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)

        # save data
        if i % 50 == 0 or i == kwargs['epoch'] - 1:
            with torch.no_grad():
                write_curve_points(f"{output_path}/training_save/{i}_curve.obj", curves[0], CTRL_P_NUM)
                write_obj(f"{output_path}/training_save/{i}_mesh.obj", patches[0], CTRL_P_NUM)

                '''
                # save every patch
                if i == kwargs['epoch'] - 1:
                    for j, patch in enumerate(patches[0]):
                        write_obj(f"{output_path}/training_save/patch{j}.obj", patch.unsqueeze(0), CTRL_P_NUM)
                '''

                # curve probability
                # curves (b, curve_num, cp_num, 3)
                curve_sample_num = 16
                curves, curves_mask = curve_probability(prep_points, curves[0], curve_sample_num)
                torch.save(curves_mask, f'{output_path}/curves_mask.pt')
                # write_curve_points(f"{output_path}/training_save/{i}_curve_d.obj", curves, CTRL_P_NUM)

                # save control points
                save_obj(f"{output_path}/control_points.obj", cp_coord[0])

                # save nn model
                torch.save(model, f"{output_path}/model_weights.pth")

                # save pcd
                save_obj(f"{output_path}/training_save/pcd.obj", pcd_points)
                
    # save loss and lr logs as images
    log_save_dir = os.path.join(output_path, "logs")
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir, exist_ok=True)
    save_loss_fig(loss_list, log_save_dir) # save loss image
    save_lr_fig(lr_list, log_save_dir) # save learning rate image

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="outputs/cat")
    parser.add_argument('--prep_output_path', type=str, default="outputs/cat/prep_outputs/train_outputs")

    parser.add_argument('--epoch', type=int, default="201")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.0002")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()

    training(**vars(args))