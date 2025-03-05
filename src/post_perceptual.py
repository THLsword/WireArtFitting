import os
import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
import networkx as nx
from scipy.interpolate import BSpline
import alphashape
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
from tqdm import tqdm
import time

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability
from utils.postprocess_utils import get_unique_curve, project_curve_to_pcd, delete_single_curve, create_curve_graph, find_deletable_edges, compute_IOU
from utils.create_mesh import create_mesh
from utils.graph_utils import minimum_path_coverage
from utils.save_data import save_img, save_obj, save_curves

from model.model_interface import Model

def create_bspline(mean_curve_points):
    print("mean_curve_points:", mean_curve_points.shape)
    k = 3
    sample_num = 50
    curves = np.zeros([mean_curve_points.shape[0], sample_num, 3])
    for i, curve in enumerate(mean_curve_points):
        # curve (n, 3) 
        # rangeeduce number of sampling points
        head = curve[0, :]
        tail = curve[-1, :]
        middle = curve[1:-1:2, :]
        curve = torch.vstack((head, middle, tail))
        # bspline
        curve = curve.numpy()
        _, idxs = np.unique(curve, axis=0, return_index=True)
        unique_curve = curve[np.sort(idxs)]
        cp_num = unique_curve.shape[0]
        k = min(3,cp_num-1)
        m = k + cp_num + 1
        t = np.linspace(0, 1, m-2*k)
        t = np.concatenate(([t[0]]*k, t, [t[-1]]*k))
        spl = BSpline(t, unique_curve, k)
        tnew = np.linspace(t[0], t[-1], sample_num)
        new_curve = spl(tnew) # (sample_num, 3)
        curves[i] = new_curve
    return curves

def render(data):
    """
    render curves and compute alphashape  
    inputs: dict ['bspline_remian', 'image_size', 'i: int', 'alpha_value', 'save_img: bool']  
    outputs: alphashape area and length
    alpha_value越小，輪廓越大（細節越差）
    """
    rotated_pcd = data['bspline_remian']
    image_size = data['image_size']
    num = data['i'] # for saving png
    alpha_value = data['alpha_value']
    save_img =  data['save_img']

    # projection and image
    projection = rotated_pcd[:, 1:3]

    img = np.zeros((image_size, image_size))
    for point in projection:
        x_idx = int(point[0] * (image_size - 1)/2)+int(image_size/2)
        y_idx = int(point[1] * (image_size - 1)/2)+int(image_size/2)
        img[x_idx, y_idx] = 1
        if y_idx+1 < image_size and x_idx+1 < image_size:
            img[x_idx+1, y_idx+1] = 1 
    del projection

    # compute area
    y, x = np.where(img > 0)
    points_2d = list(zip(x, y))
    alpha_shape_pcd = alphashape.alphashape(points_2d, alpha_value)
    area = alpha_shape_pcd.area
    length = alpha_shape_pcd.length

    if save_img:
        output_path = "render_results"
        os.makedirs(output_path, exist_ok=True)
        # test & save image
        image = Image.fromarray((img*255).astype(np.uint8))
        image.save(f"{output_path}/render{num}.png")

        # save polygon image
        image = Image.new("1", (image_size,image_size), 0)  # "1"表示二值化模式，0表示黑色
        draw = ImageDraw.Draw(image)
        draw.polygon(list(alpha_shape_pcd.exterior.coords), outline=1, fill=1)
        image.save(f"{output_path}/polygon{num}.png")

    return area, length

def create_graph(curve_idx):
    curve_idx = np.array(curve_idx)
    G = nx.Graph()
    # add edges into G
    for i, curve in enumerate(curve_idx):
        G.add_edge(curve[0],curve[-1])
        G.edges[curve[0],curve[-1]]['idx']=[i]
    return G

def graph_delete_curve(G, idx):
    G_ = G.copy()
    # idx is the curve's idx
    for edge in list(G_.edges):
        if G_.edges[edge[0],edge[1]]['idx'] == idx:
            G_.remove_edge(edge[0], edge[1])
            break
    # # if node's degree=2, merge 2 curves
    # for node in list(G_.nodes):
    #     if G_.degree(node) == 2:
    #         for i,value in G_.adjacency():
    #             if i==node:
    #                 adj = []    # len=2
    #                 adj_idx = []# len=2
    #                 for j in value:
    #                     adj.append(j)
    #                     for k in value[j]['idx']:
    #                         adj_idx.append(k)
    #                 for j in adj:
    #                     G_.remove_edge(j,i)
    #                 G_.add_edge(adj[0],adj[1])
    #                 G_.edges[adj[0],adj[1]]['idx']=adj_idx
    for node in list(G_.nodes):
        if G_.degree[node] == 0:
            G_.remove_node(node)
    return G_

def graph_curve_removable(G, idx):
    G_ = G.copy()
    # # detect if one degree is 3
    # for edge in list(G_.edges):
    #     if G_.edges[edge[0],edge[1]]['idx'] == idx:
    #         if G_.degree[edge[0]]==3 or G_.degree[edge[1]]==3:
    #             break
    #         else:
    #             return False
    # detect if delete this curve, will cause cutting
    for edge in list(G_.edges):
        if G_.edges[edge[0],edge[1]]['idx'] == idx:
            G_.remove_edge(edge[0], edge[1])
            break
    # for edge in list(G_.edges):
    #     if G_.degree(edge[0]) == 1 or G_.degree(edge[1]) == 1:
    #         return False
    for node in list(G_.nodes):
        if G_.degree[node] == 0:
            G_.remove_node(node)
    g_num = len(list(nx.connected_components(G_)))
    if g_num > 1:
        return False
    else:
        return True

def training(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    # load data
        # load .npz point cloud
    model_path = kwargs['model_path']
    output_path = kwargs['output_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    # pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_maen = abs(pcd_points).mean(0) # [3]

    # load preprocessing' weights .pt
    prep_output_path = kwargs['prep_output_path']
    weight_path = os.path.join(prep_output_path, 'weights.pt')
    if os.path.exists(weight_path):
        mview_weights_ = torch.load(weight_path)
        # 0~1 -> -0.25~0.25
        mview_weights = ((mview_weights_ - 0.5)/3).detach()
        mview_weights.requires_grad_(False)
    else:
        print(f"{weight_path} doesn't exist, mview_weights = None")
        mview_weights = None

    # multi-view points sample from pcd
    mv_mask = mview_weights_>0.5
    mv_points = multiview_sample(pcd_points, mview_weights_) # (M, 3)

    # load template
    batch_size = kwargs['batch_size']
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    # template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx =\
    # template_params.to(device), vertex_idx.to(device), face_idx.to(device), symmetriy_idx.to(device), curve_idx.to(device)
    
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))
    # resize template(to be similar in size)
    template_mean = abs(template_params.view(-1,3)).mean(0) # [3]
    template_params = (template_params.view(-1,3) / template_mean * pcd_maen)
    template_params = template_params.repeat(batch_size, 1, 1)

    # unique curves: curves were calculated twice
    curve_idx = get_unique_curve(curve_idx)
    G = create_graph(curve_idx)

    # load control points -> curves 
    output_path = kwargs['output_path']
    control_points = load_obj(f"{output_path}/control_points.obj")
    curves = control_points[curve_idx] # (curve_num, cp_num, 3)
    # delete useless curve or not
    if os.path.exists(f'{output_path}/curves_mask.pt') and kwargs['d_curve']:
        curves_mask = torch.load(f'{output_path}/curves_mask.pt')
        curves = curves[curves_mask]

    # load nn model
    output_path = kwargs['output_path']
    model = torch.load(f"{output_path}/model_weights.pth").to(device)

    ################### post-porcessing #################
    ################### post-porcessing #################
    # project_curve_to_pcd
    '''  (n, 3) only used to save as obj
    review_idx :     (curve_num, 140, k) index of pcd 
    curve_idx_list : (curve_num, n) each curve's unique idx of pcd
    curve_cood_list: (curve_num, n, 3)
    '''
    sampled_pcd, review_idx, curve_idx_list, curve_cood_list = project_curve_to_pcd(curves, pcd_points.repeat(batch_size,1,1), batch_size, sample_num, kwargs['k'])
    print("project to point clod")
    save_obj(f"{output_path}/sampled_pcd.obj", sampled_pcd)

    # get pcd's PCA, Bsplines, and determine different views
    pcd_np = np.array(pcd_points)
    pca = PCA(n_components=3)
    pca.fit(pcd_np)
    all_bspline = create_bspline(pcd_points[review_idx].mean(dim=2)) # (48, 400, 3)
    transformed_data = pca.transform(pcd_np)
    transformed_bspline = pca.transform(np.array(all_bspline).reshape((-1,3)))
    transformed_bspline = transformed_bspline.reshape((all_bspline.shape))
    pca_x, pca_y, pca_z = np.max(transformed_data, axis=0) + np.abs(np.min(transformed_data, axis=0))

    # rotate matrix (will project to yz plane)
    # rotate_y_angels = [0.0, np.arctan2(pca_z*1.5, pca_x), np.pi/2, np.pi-np.arctan2(pca_z*1.5, pca_x)]
    rotate_y_angels = [-np.pi*0.33, 0.0, np.pi*0.33, np.pi/2, np.pi - np.pi*0.33]
    rotate_matrix = []
    for i in rotate_y_angels:
        matrix = np.array([
            [np.cos(i), 0, np.sin(i)],
            [0, 1, 0],
            [-np.sin(i), 0, np.cos(i)]
        ])
        rotate_matrix.append(matrix)
    # 俯視視角
    rotate_matrix.append([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    rotate_matrix = np.stack(rotate_matrix)

    # compute perceptual loss
    cross_attention = kwargs['crossattention']
    total_curve_unm = review_idx.shape[0]
    # object_curve_num = int(total_curve_unm/1.7)
    object_curve_num = kwargs['object_curve_num']

    pcd_feature = model.pcd_backbone(model.embedding_layer(pcd_points.repeat(batch_size, 1, 1).transpose(1, 2).to(device))) #[1, 3, 4096] -> [1, 128, 4096]
    mv_feature = model.prep_backbone(model.embedding_layer(mv_points.repeat(batch_size, 1, 1).transpose(1, 2).to(device)))

    if cross_attention:  
        pcd_feature = model.upsample(pcd_feature, mv_feature)
    # else:
    #     pcd_feature = model.upsample(pcd_feature, mv_feature)
    #     pcd_feature, _ = pcd_feature.max(dim = 2)
    
    # render & alphashape before remove curves
    image_size = 128
    alpha_value = kwargs['alpha_value']
    bspline_remian = all_bspline.reshape((-1,3))
        # rotate bspline
    rotated_bsplines = []
    for mat in rotate_matrix:
        transformed = bspline_remian @ mat.T
        rotated_bsplines.append(transformed)
    rotated_bsplines = np.stack(rotated_bsplines, axis=0)

    data_list = []
    for i, value in enumerate(rotated_bsplines):
        data_list.append({'bspline_remian':value, 'image_size':image_size, 'i':i, 'alpha_value':alpha_value, 'save_img':True})
    with ProcessPoolExecutor(max_workers=len(rotate_matrix)) as executor:
        as_results = list(executor.map(render, data_list)) # list [(area,length),...,(area,length)]
    area_before_delet = [] # area before deleting
    length_bd = [] # length before deleting
    for as_result in as_results:
        area_before_delet.append(as_result[0])
        length_bd.append(as_result[1])

    ## start removing curves
    mv_thresh = kwargs['mv_thresh']
    curves_ramian = curve_cood_list.copy()
    area_global = np.array(area_before_delet) # global is the value before removing
    while True:
        delete_idx = []
        L2_losses = []
        L2_losses_all = np.ones(len(curve_cood_list))*100
        min_L2_loss = 100
        for edge in tqdm(list(G.edges)):
            j = G.edges[edge[0],edge[1]]['idx'] # list

            # detect
            continue_bool = False
            if not graph_curve_removable(G, j):
                continue_bool = True
            if continue_bool:
                continue

            ## perceptual loss
            remain_idx_list = [] # idx list except j
            for k in list(G.edges):
                idx_list = G.edges[k[0],k[1]]['idx']
                if idx_list != j:
                    for l in idx_list:
                        remain_idx_list.append(l)

            curves_ramian_ = [curves_ramian[k] for k in remain_idx_list]
            curves_idx_ramian_ = [curve_idx_list[k] for k in remain_idx_list] # idx of points in each curves
            temp_points = torch.unique(torch.cat(curves_ramian_), dim=0)
            temp_idxs = torch.unique(torch.cat(curves_idx_ramian_), dim=0)

            # 對mv points求mask，刪掉的店在mv points中也同樣刪除
            mask1 = torch.zeros(pcd_points.shape[0], dtype=torch.bool)
            mask1[temp_idxs] = True
            result_mask = mask1 & mv_mask.to('cpu')
            masked_mvpoints = pcd_points[result_mask]

            # 隨機重複點，直到填滿[4096,3] & shuffle
            indices = torch.randint(0, temp_points.size(0), (pcd_points.shape[0]-temp_points.shape[0],)) 
            temp_points = torch.cat((temp_points, temp_points[indices]), dim=0)
            indices = torch.randperm(temp_points.size(0))  # 生成一个从 0 到 4095 的随机排列索引
            temp_points = temp_points[indices]
            noise = torch.normal(mean=0.0, std=0.001, size=temp_points.shape)
            temp_points = temp_points + noise

            # 隨機重複點，填滿masked mv points
            indices = torch.randint(0, masked_mvpoints.size(0), (mv_points.shape[0]-masked_mvpoints.shape[0],)) 
            masked_mvpoints = torch.cat((masked_mvpoints, masked_mvpoints[indices]), dim=0)
            indices = torch.randperm(masked_mvpoints.size(0))  # 生成一个从 0 到 4095 的随机排列索引
            masked_mvpoints = masked_mvpoints[indices]

            # model inference
            samplepcd_feature = model.pcd_backbone(model.embedding_layer(temp_points.repeat(batch_size, 1, 1).transpose(1, 2).to(device)))
            samplepcd_mv_feature = model.prep_backbone(model.embedding_layer(masked_mvpoints.repeat(batch_size, 1, 1).transpose(1, 2).to(device)))
            if cross_attention: 
                samplepcd_feature = model.upsample(samplepcd_feature, samplepcd_mv_feature)
            # else:
            #     samplepcd_feature, _ = samplepcd_feature.max(dim = 2)

            with torch.no_grad():
                L2_loss = F.mse_loss(samplepcd_feature, pcd_feature)
                if L2_loss < min_L2_loss:
                    min_L2_loss = L2_loss
                    delete_idx = j
                L2_losses.append(L2_loss.to('cpu'))
                for idx_temp in j:                    # 注釋這行，跳過計算IOU
                    L2_losses_all[idx_temp] = L2_loss # 注釋這行，跳過計算IOU

        if len(L2_losses) == 0:
            print("no curve can be remove")
            break
        L2_losses_np = np.array(L2_losses)
        sorted_arr = np.sort(L2_losses_np)
        half_value = sorted_arr[len(sorted_arr) // 2]
        L2_losses_bool = L2_losses_all < half_value

        min_IOU_loss = 1
        for edge in tqdm(list(G.edges)):
            j = G.edges[edge[0],edge[1]]['idx'] # list
            # detect
            continue_bool = False
            if not graph_curve_removable(G, j):
                continue_bool = True
            for idx_temp in j:
                if not L2_losses_bool[idx_temp]:
                    continue_bool = True
            if continue_bool:
                continue
            
            ## IOU area/length after deletion
            remain_idx_list = [] # idx list except j
            for k in list(G.edges):
                idx_list = G.edges[k[0],k[1]]['idx']
                if idx_list != j:
                    for l in idx_list:
                        remain_idx_list.append(l)
            # bspline_remian = transformed_bspline[remain_idx_list].reshape((-1,3))
            bspline_remian = all_bspline[remain_idx_list].reshape((-1,3))
            data_list = []
            area_ad   = []
            length_ad = []
            for i, value in enumerate(rotate_matrix):
                data_list.append({'bspline_remian':bspline_remian, 'rotate_matrix':value, 'image_size':image_size, 'i':i, 'alpha_value':alpha_value, 'save_img':False})
            with ProcessPoolExecutor(max_workers=len(rotate_matrix)) as executor:
                as_results = list(executor.map(render, data_list))
            for as_result in as_results:
                area_ad.append(as_result[0])
                length_ad.append(as_result[1])

            IOU_loss_global = max((area_global-np.array(area_ad))/area_global)
            if IOU_loss_global < min_IOU_loss:
                min_IOU_loss = IOU_loss_global
                delete_idx = j

        if min_IOU_loss > mv_thresh and min_IOU_loss != 1:
            print("min_IOU_loss:", min_IOU_loss," > mv_thresh")
            break
        # graph delete & losses
        G = graph_delete_curve(G, delete_idx)
        print("min IOU loss: ", min_IOU_loss)
        # break condition
        remain_idx_list = []
        for k in list(G.edges):
            idx_list = G.edges[k[0],k[1]]['idx']
            for l in idx_list:
                remain_idx_list.append(l)
        print("curves remain : ", len(remain_idx_list))
        
        if len(remain_idx_list) <= object_curve_num:
            print("len(remain_idx_list) <= object_curve_num")
            break

    remain_idx_list = []
    for k in list(G.edges):
        idx_list = G.edges[k[0],k[1]]['idx']
        for l in idx_list:
            remain_idx_list.append(l)
    curves_ramian_tensor = torch.cat([curves_ramian[k] for k in remain_idx_list])
    bspline_remian = all_bspline[remain_idx_list].reshape((-1,3))
    print('object_curve_num: ', object_curve_num)
    create_mesh(all_bspline[remain_idx_list], 0.003, output_path)
    save_obj(f"{output_path}/sampled_pcd_perceptual.obj", curves_ramian_tensor)
    save_obj(f"{output_path}/spline_perceptual.obj", bspline_remian)
        
if __name__ == '__main__':
    start_time = time.time() 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="outputs/cat")
    parser.add_argument('--prep_output_path', type=str, default="outputs/cat/prep_outputs/train_outputs")

    parser.add_argument('--epoch', type=int, default="201")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.0005")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=10) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 
    parser.add_argument('--alpha_value', type=float, default=0.2) 
    parser.add_argument('--object_curve_num', type=float, default=25) 
    parser.add_argument('--mv_thresh', type=float, default=0.10)
    parser.add_argument('--crossattention', type=bool, default=True)

    args = parser.parse_args()
    training(**vars(args))
    end_time = time.time()
    print(f"time: {(end_time - start_time):.4f}")
