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
from scipy.spatial import ConvexHull

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability
from utils.save_data import save_img, save_obj, save_lr_fig, save_loss_fig
from utils.convexhull_utils import sample_convex_hull_with_normals
from utils.data_augmentation_utils import add_gaussian_noise, random_rotate_point_cloud

from model.model_interface import Model

def lr_lambda(epoch, warm_epoch):
    if epoch < warm_epoch:
        # 線性 warm up：從較小值增長到 1
        # return float(epoch + 1) / warm_epoch
        return 0.1
    else:
        # 從 warm up 結束後開始衰減
        decay_rate = 0.97  # 衰減因子，可根據需求調整
        return decay_rate ** (epoch - warm_epoch)


def compute_loss(cp_coord, patches, curves, pcd_points, pcd_normals, pcd_area, prep_points, sample_num, tpl_sym_idx, prerp_weights_scaled, i: int, epoch_num: int, warm_epoch: int):
    batch_size = patches.shape[0]
    face_num = patches.shape[1]
    st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(patches).float() # [b, patch_num, sample_num, 2]

    chamfer_weight_rate = 0.5

    # preprocessing
    # patches (B,face_num,cp_num,3)
    points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
    normals = coons_normals(st[..., 0], st[..., 1], patches)
    mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 
    # pcd_points  = pcd_points.repeat(batch_size,1,1)
    # prep_points = prep_points.repeat(batch_size,1,1)
    # pcd_normals = pcd_normals.repeat(batch_size,1,1)
    pcd_area    = pcd_area.repeat(batch_size,1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss, normal_loss = area_weighted_chamfer_loss(
        mtds, points, normals, pcd_points, pcd_normals, chamfer_weight_rate, prerp_weights_scaled
    )

    # curve chamfer loss
    # curves (b, curve_num, cp_num, 3)
    curves_s_num = 16
    linspace = torch.linspace(0, 1, curves_s_num).to(curves).flatten()[..., None].float()
    curve_points = bezier_sample(linspace, curves)
    # curve_chamfer_loss, _, _, _ = curve_chamfer(curve_points, pcd_points*1.01)
    # curve_chamfer_loss = curve_chamfer_loss.mean(1)

    # # multi view curve loss
    # _, _, mv_curve_loss, _ = curve_chamfer(curve_points, prep_points)
    # mv_curve_loss = mv_curve_loss.mean(1)

    # flatness loss
    planar_loss = planar_patch_loss(st, points, mtds)

    # Conciseness loss
    overlap_loss = patch_overlap_loss(mtds, pcd_area)

    # Orthogonality loss
    perpendicular_loss = curve_perpendicular_loss(patches)

    # flatness loss
    FA_loss = flatness_area_loss(st, points, mtds)

    # symmetry loss
    symmetry_loss = torch.zeros(1).to(patches).float()
    symmetry_loss = patch_symmetry_loss(tpl_sym_idx[0], tpl_sym_idx[1], cp_coord)

    # curvature loss
    curvature_loss = curve_curvature_loss(curves, linspace)

    # beam gap loss
    thres = 0.8 + (0.1 * i / epoch_num)
    beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, thres)

    # normal loss &  beam gap loss weight
    e_ = warm_epoch
    if i <= e_: 
        normal_loss = 0.1 * normal_loss * math.exp((i-e_)/e_)
        # beam_gap_loss = 0 * beam_gap_loss * math.exp((i-e_)/e_)
    else:
        normal_loss = 0.1 * normal_loss

    # test loss
    # stable_loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss + normal_loss + 0.2 * beam_gap_loss
    stable_loss = chamfer_loss + 2.5 *planar_loss + 0.1*symmetry_loss + 0.8 *normal_loss + 0.5 * beam_gap_loss# + 0.002 * curvature_loss# + 0.005*overlap_loss
    loss = stable_loss

    loss_dict = {
        'chamfer_loss': chamfer_loss.item(),
        'normal_loss': normal_loss.item(),
        'planar_loss': planar_loss.mean().item(),
        'symmetry_loss': symmetry_loss.item(),
        'beam_gap_loss': beam_gap_loss.item(),
    }

    return loss.mean(), loss_dict

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
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device).float(), pcd_normals.to(device).float(), pcd_area.to(device).float()
    pcd_maen = abs(pcd_points).mean(0) # [3]

    # load preprocessing output: 'weights.pt'
    prep_output_path = kwargs['prep_output_path']
    weight_path = os.path.join(prep_output_path, 'weights.pt')
    if os.path.exists(weight_path):
        prep_weights = torch.load(weight_path)
        # 0~1 -> -0.25~0.25
        prerp_weights_scaled = ((prep_weights - 0.5)/2).detach().float()
        prerp_weights_scaled.requires_grad_(False)
    else:
        print(f"{weight_path} doesn't exist, prerp_weights_scaled = None")
        prerp_weights_scaled = None
    # multi-view points sample from pcd
    prep_points = multiview_sample(pcd_points, prep_weights).float() # (M, 3)

    # load template
    template_path = kwargs['template_path']
    tpl_params, tpl_v_idx, tpl_f_idx, tpl_sym_idx, tpl_c_idx = load_template(template_path)
    tpl_params, tpl_v_idx, tpl_f_idx, tpl_sym_idx, tpl_c_idx =\
    tpl_params.to(device).float(), tpl_v_idx.to(device), tpl_f_idx.to(device), tpl_sym_idx.to(device), tpl_c_idx.to(device)
        # Number of samples on one face of the template
    sample_num = int(np.ceil(np.sqrt(pcd_points.shape[0] / tpl_f_idx.shape[0])))
        # Scale the template to a size similar to that of the point cloud
    template_mean = abs(tpl_params.view(-1,3)).mean(0) # [3]
    tpl_params = (tpl_params.view(-1,3) / template_mean * pcd_maen).float()

    # 計算convex hull
    cx_hull_points, cx_hull_normals = sample_convex_hull_with_normals(pcd_points.cpu().numpy(), num_samples=4096)
    cx_hull_points = torch.tensor(cx_hull_points).to(device).float() # 将 cx_hull_points 转换为 tensor 并移动到 GPU
    cx_hull_normals = torch.tensor(cx_hull_normals).to(device).float() # 将 cx_hull_normals 转换为 tensor 并移动到 GPU

    # train
    BATCH_SIZE = kwargs['batch_size']
        # data augmentation: 旋轉 + 高斯噪聲
    augmented_pcd_points, rotation_matrices = random_rotate_point_cloud(pcd_points.cpu().numpy(), BATCH_SIZE, 0)
    augmented_pcd_points = add_gaussian_noise(augmented_pcd_points, noise_std=0.000) # (B, 4096, 3)
    augmented_pcd_normals = np.matmul(np.expand_dims(pcd_normals.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1)) # (B, 4096, 3)
    augmented_cx_hull_points = np.matmul(np.expand_dims(cx_hull_points.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1))  # (B, 4096, 3)
    augmented_cx_hull_normals = np.matmul(np.expand_dims(cx_hull_normals.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1))  # (B, 4096, 3)
    augmented_prep_points= np.matmul(np.expand_dims(prep_points.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1)) # (B, M, 3)
        # to tensor & to device
    augmented_pcd_points = torch.tensor(augmented_pcd_points).to(device).float() # (B, 4096, 3)
    augmented_pcd_normals = torch.tensor(augmented_pcd_normals).to(device).float() # (B, 4096, 3)
    augmented_cx_hull_points = torch.tensor(augmented_cx_hull_points).to(device).float() # (B, 4096, 3)
    augmented_cx_hull_normals = torch.tensor(augmented_cx_hull_normals).to(device).float() # (B, 4096, 3)
    augmented_prep_points = torch.tensor(augmented_prep_points).to(device).float() # (B, M, 3)
        # 旋轉template
    tpl_params_rotated = []
    for i in range(BATCH_SIZE):
        tpl_params_rotated.append(np.dot(tpl_params.cpu().numpy(), rotation_matrices[i].T))
    tpl_params_rotated = torch.tensor(np.stack(tpl_params_rotated, axis=0)).to(device).float() # (B, N, 3)

    CTRL_P_NUM = 4 # number of control points on one curve
    # model = Model(tpl_params_rotated).cuda()
    model = Model(tpl_params.repeat(BATCH_SIZE, 1, 1)).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'], betas=(0.9, 0.99))
    loop = tqdm(range(kwargs['epoch']))
    warm_epoch = kwargs['warm_epoch']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, warm_epoch))

    loss_list = []
    lr_list = []
    loss_dict_list = []
    for i in loop:
        # # data augmentation: 旋轉 + 高斯噪聲
        # augmented_pcd_points, rotation_matrices = random_rotate_point_cloud(pcd_points.cpu().numpy(), BATCH_SIZE)
        # augmented_pcd_points = add_gaussian_noise(augmented_pcd_points, noise_std=0.005) # (B, 4096, 3)
        # augmented_pcd_normals = np.matmul(np.expand_dims(pcd_normals.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1)) # (B, 4096, 3)
        # augmented_cx_hull_points = np.matmul(np.expand_dims(cx_hull_points.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1))  # (B, 4096, 3)
        # augmented_cx_hull_normals = np.matmul(np.expand_dims(cx_hull_normals.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1))  # (B, 4096, 3)
        # augmented_prep_points= np.matmul(np.expand_dims(prep_points.cpu().numpy(), axis=0), rotation_matrices.transpose(0, 2, 1)) # (B, M, 3)
        #     # to tensor & to device
        # augmented_pcd_points = torch.tensor(augmented_pcd_points).to(device).float() # (B, 4096, 3)
        # augmented_pcd_normals = torch.tensor(augmented_pcd_normals).to(device).float() # (B, 4096, 3)
        # augmented_cx_hull_points = torch.tensor(augmented_cx_hull_points).to(device).float() # (B, 4096, 3)
        # augmented_cx_hull_normals = torch.tensor(augmented_cx_hull_normals).to(device).float() # (B, 4096, 3)
        # augmented_prep_points = torch.tensor(augmented_prep_points).to(device).float() # (B, M, 3)

        ## 舊的打亂索引
        # indices = torch.randperm(pcd_points.size(0))  # 生成一个从 0 到 1023 的随机排列索引
        # shuffled_pcd_points = pcd_points[indices]

        # 打亂augmented_pcd_points的索引
        b, n, _ = augmented_pcd_points.shape
            # 為每個 batch 生成隨機索引，indices 的形狀為 [b, n]
        indices = torch.stack([torch.randperm(n) for _ in range(b)], dim=0)
            # torch.arange(b) 生成 [0, 1, ..., b-1]，並通過 unsqueeze 變成 [b, 1]，
            # 與 indices 搭配實現對每個 batch 的第二維進行索引
        shuffled_pcd_points = augmented_pcd_points[torch.arange(b).unsqueeze(1), indices]

        cp_coord = model(shuffled_pcd_points, augmented_prep_points) # MLP:(B, N), review to (B, -1, 3)
        patches = cp_coord[:,tpl_f_idx] # (B, face_num, cp_num, 3)
        curves = cp_coord[:,tpl_c_idx] # (B, curve_num, cp_num, 3)
        prerp_weights_scaled = 1 + (prerp_weights_scaled/2).float() # 0.875~1.125
        if i <= warm_epoch:
            # 在warm up階段，gt替換為convex hull
            loss, loss_dict = compute_loss(
                cp_coord=cp_coord, # [B, n, 3]
                patches=patches,
                curves=curves,
                # pcd_points=augmented_cx_hull_points, # (B, 4096, 3)
                # pcd_normals=augmented_cx_hull_normals, # (B, 4096, 3)
                pcd_points=augmented_pcd_points,
                pcd_normals=augmented_pcd_normals,
                pcd_area=pcd_area,
                prep_points=prep_points,
                sample_num=sample_num,
                tpl_sym_idx=tpl_sym_idx,
                prerp_weights_scaled=None,
                i=i,
                epoch_num=kwargs['epoch'],
                warm_epoch=warm_epoch
            )
        else:
            loss, loss_dict = compute_loss(
                cp_coord=cp_coord, # [B, n, 3]
                patches=patches,
                curves=curves,
                pcd_points=augmented_pcd_points,
                pcd_normals=augmented_pcd_normals,
                pcd_area=pcd_area,
                prep_points=prep_points,
                sample_num=sample_num,
                tpl_sym_idx=tpl_sym_idx,
                prerp_weights_scaled=prerp_weights_scaled.to(device),
                i=i,
                epoch_num=kwargs['epoch'],
                warm_epoch=warm_epoch
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
            loss_dict_list.append(loss_dict)

        # save data
        if i % 50 == 0 or i == kwargs['epoch'] - 1 or i == warm_epoch:
            with torch.no_grad():
                write_curve_points(f"{output_path}/training_save/{i}_curve.obj", curves[0], CTRL_P_NUM)
                for j in range(BATCH_SIZE):
                    write_obj(f"{output_path}/training_save/{i}_mesh{j}.obj", patches[j], CTRL_P_NUM)
                # write_obj(f"{output_path}/training_save/{i}_mesh.obj", patches[0], CTRL_P_NUM)

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

    # save individual loss logs as images
    for key in loss_dict_list[0].keys():
        save_loss_fig([d[key] for d in loss_dict_list], log_save_dir, key) # save individual loss image

    # eval
    model.eval()
    cp_coord = model(pcd_points.repeat(1,1,1), prep_points.repeat(1,1,1))
    with torch.no_grad():
        patches = cp_coord[:,tpl_f_idx] # (B, face_num, cp_num, 3)
        curves = cp_coord[:,tpl_c_idx] # (B, curve_num, cp_num, 3)
        write_curve_points(f"{output_path}/eval_curve.obj", curves[0], CTRL_P_NUM)
        write_obj(f"{output_path}/eval_mesh.obj", patches[0], CTRL_P_NUM)

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="outputs/cat")
    parser.add_argument('--prep_output_path', type=str, default="outputs/cat/prep_outputs/train_outputs")

    parser.add_argument('--epoch', type=int, default="201")
    parser.add_argument('--batch_size', type=int, default="1") 
    parser.add_argument('--learning_rate', type=float, default="0.0001")
    parser.add_argument('--warm_epoch', type=int, default=50)

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()

    training(**vars(args))