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
from PIL import Image
from sklearn.decomposition import PCA

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability

from model.backbone.apes_seg_backbone import APESSeg2Backbone
from model.backbone.apes_cls_backbone import APESClsBackbone, simple_mlp
from model.head.apes_cls_head import MLP_Head
from model.utils.layers import UpSample

class Model_warmup(nn.Module):
    def __init__(self, template_params, batch_size):
        super(Model_warmup, self).__init__()
        self.batch_size = batch_size
        # template_params input shape : [B, N, C]
        self.register_buffer('template_params', template_params[0].reshape(-1)) # [N*C]

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_params)))

    def forward(self):
        vertices = self.template_params + self.displace # [N*C]
        return rearrange(vertices, '(N C) -> N C',C=3)

class Model(nn.Module):
    def __init__(self, template_params):
        super(Model, self).__init__()
        self.register_buffer('template_params', template_params) # (B, 122, 3)
        self.cp_num = self.template_params.shape[1] * self.template_params.shape[2]

        self.simple_MLP  = simple_mlp()
        self.simple_MLP2 = simple_mlp() 
        self.upsample = UpSample()

        self.head = MLP_Head()

    def forward(self, pcd, mv_points):
        pcd = torch.transpose(pcd, 1, 2) # (B, N, 3) -> (B, 3, N)
        mv_points = torch.transpose(mv_points, 1, 2) # (B, M, 3) -> (B, 3, M)

        pcd_feature = self.simple_MLP(pcd) # (B, C, N)
        output = self.head(pcd_feature) # (B, C, N) -> (B, 366)

        ds_feature = self.simple_MLP2(mv_points) # (B, C, N)
        temp = self.upsample(pcd_feature, ds_feature) # (B, C, N)
        output = self.head(temp)
        
        output = rearrange(output, 'B (M N) -> B M N', N = 3)
        output = self.template_params + output

        return output

def lr_lambda(epoch):
    warm_epoch = 20
    if epoch < warm_epoch:
        return (epoch + 1) / warm_epoch
    else:
        return 0.99 ** (epoch - warm_epoch)

def compute_warm_up_loss(vertices, patches, pcd_points, sample_num, symmetriy_idx):
    batch_size = patches.shape[0]
    face_num = patches.shape[1]
    st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(patches) # [b, patch_num, sample_num, 2]

    # preprocessing
    # patches (B,face_num,cp_num,3)
    points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
    normals = coons_normals(st[..., 0], st[..., 1], patches)
    mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 
    pcd_points  = pcd_points.repeat(batch_size,1,1)

    # area-weighted chamfer loss (position + normal)
    chamfer_loss = warm_up_chamfer_loss(mtds, points, pcd_points)

    # flatness loss
    planar_loss = planar_patch_loss(st, points, mtds)

    # symmetry loss
    symmetry_loss = torch.zeros(1).to(patches)
    symmetry_loss = patch_symmetry_loss(symmetriy_idx[0], symmetriy_idx[1], vertices)

    loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss

    return loss.mean()

def compute_loss(vertices, patches, curves, pcd_points, pcd_normals, pcd_area, mv_points, sample_num, symmetriy_idx, multi_view_weights, i, epoch_num):
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
    curve_chamfer_loss, _, _, _ = curve_chamfer(curve_points, pcd_points)
    curve_chamfer_loss = curve_chamfer_loss.mean(1)

    # multi view curve loss
    _, _, mv_curve_loss, _ = curve_chamfer(curve_points, mv_points)
    mv_curve_loss = mv_curve_loss.mean(1)

    # flatness loss
    planar_loss = planar_patch_loss(st, points, mtds)

    # # Conciseness loss
    # overlap_loss = patch_overlap_loss(mtds, pcd_area)

    # Orthogonality loss
    perpendicular_loss = curve_perpendicular_loss(patches)

    # flatness loss
    FA_loss = flatness_area_loss(st, points, mtds)

    # symmetry loss
    symmetry_loss = torch.zeros(1).to(patches)
    symmetry_loss = patch_symmetry_loss(symmetriy_idx[0], symmetriy_idx[1], vertices)

    # # curvature loss
    # curvature_loss = curve_curvature_loss(curves, linspace)

    # beam gap loss
    thres = 0.9
    beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, thres)
    
    # # loss = chamfer_loss + 0.01*overlap_loss + 2*planar_loss + 0.1*symmetry_loss
    # stable_loss = chamfer_loss + 0.01*overlap_loss + 2.5*planar_loss + 0.1*symmetry_loss# + 0.012 * curvature_loss
    # if i <= 50:
    #     loss = stable_loss + 0.012 * curvature_loss
    #             # + 0.1 * math.exp(-i/100) * normal_loss
    #             #+ 0.05 * mv_curve_loss 
    # elif i > 50 and i <= 100:
    #     loss = stable_loss + 0.012 * curvature_loss + 0.1*normal_loss * math.exp((i-150)/100)
    # else :
    #     loss = stable_loss* math.exp((100-i)/100) +  0.1*normal_loss * math.exp((i-150)/100) + 0.01*curvature_loss * math.exp((100-i)/100)
    #             #+ math.exp((i-450)/100) * curve_chamfer_loss
    #             #+ 0.05 * mv_curve_loss \

    # test loss
    stable_loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss + normal_loss + 0.2 * beam_gap_loss# + 0.012 * curvature_loss
    loss = stable_loss

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

def model2ellipsoid(data):
    data = data.to('cpu')
    data = np.array(data)
    # PCA
    pca = PCA(n_components=3)
    pca.fit(data)
    transformed_data = pca.transform(data)
    
    # 获取主成分方向
    components = pca.components_
    singular_values = pca.singular_values_
    explained_variance = pca.explained_variance_
    mean_transformed = np.mean(transformed_data, axis=0)

    # x,y,z length in transformed data
    xyz_length = (abs(np.max(transformed_data, axis=0)) + abs(np.min(transformed_data, axis=0)))/8*3

    # ellipsoid
    num_samples = 4096
    phi = np.random.uniform(0, np.pi, num_samples)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    unit_sphere_points = np.vstack((x, y, z)).T
    ellipsoid_points = unit_sphere_points * xyz_length + mean_transformed
    ellipsoid_points_original_space  = pca.inverse_transform(ellipsoid_points)

    return torch.tensor(ellipsoid_points_original_space, dtype=torch.float32)

def training(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # load .npz point cloud
    model_path = kwargs['model_path']
    output_path = kwargs['output_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_maen = abs(pcd_points).mean(0) # [3]
    print('louad data')
    print("pcd_points shape: ", pcd_points.shape)
    print("pcd_normals shape: ", pcd_normals.shape)

    # model -> ellipsoid
    ellipsoid = model2ellipsoid(pcd_points).to(device)


    # load multi_view weights .pt
    mv_path = kwargs['mv_path']
    weight_path = os.path.join(mv_path, 'weights.pt')
    if os.path.exists(weight_path):
        mview_weights_ = torch.load(f'{model_path}/weights.pt')
        # 0~1 -> -0.25~0.25
        mview_weights = ((mview_weights_ - 0.5)/3).detach()
        mview_weights.requires_grad_(False)
    else:
        print(f"{weight_path} doesn't exist, mview_weights = None")
        mview_weights = None

    # multi-view points sample from pcd
    mv_points = multiview_sample(pcd_points, mview_weights_) # (M, 3)

    # load template
    batch_size = kwargs['batch_size']
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx =\
    template_params.to(device), vertex_idx.to(device), face_idx.to(device), symmetriy_idx.to(device), curve_idx.to(device)

    print(template_params.shape)
    print(vertex_idx.shape)
    print(face_idx.shape)
    
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))
    # resize template(to be similar in size)
    template_mean = abs(template_params.view(-1,3)).mean(0) # [3]
    template_params = (template_params.view(-1,3) / template_mean * pcd_maen)
    template_params = template_params.repeat(batch_size, 1, 1)

    # train
    batch_size = kwargs['batch_size']
    control_point_num = 4
    model_warmup = Model_warmup(template_params, batch_size).cuda()
    optimizer = torch.optim.Adam(model_warmup.parameters(), 0.02, betas=(0.5, 0.99))

    # # warm up (template preprocessing)
    # for i in tqdm.tqdm(range(10)):
    #     vertices = model_warmup() # model is changed!!!!!
    #     vertices = vertices.repeat(batch_size,1,1)
    #     patches = vertices[:,face_idx] # (B, face_num, cp_num, 3)
    #     curves = vertices[:,curve_idx] # (B, curve_num, cp_num, 3)
    #     loss_params = {
    #         'vertices':vertices,
    #         'patches':patches,
    #         'pcd_points':ellipsoid,
    #         'sample_num':sample_num,
    #         'symmetriy_idx':symmetriy_idx,
    #     }
    #     loss = compute_warm_up_loss(**loss_params)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if i == 9:
    #         with torch.no_grad():
    #             template_params = vertices
    #             os.makedirs(output_path, exist_ok=True)
    #             os.makedirs(f"{output_path}/training_save", exist_ok=True)
    #             save_obj(f"{output_path}/training_save/pcd.obj", pcd_points)
    #             write_obj(f"{output_path}/training_save/warm_up.obj", patches[0], control_point_num)
    #             save_obj(f"{output_path}/training_save/ellipsoid.obj", ellipsoid)

    model = Model(template_params).cuda()
    optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'], betas=(0.5, 0.99))
    loop = tqdm.tqdm(list(range(0, kwargs['epoch'])))
    scheduler = LambdaLR(optimizer, lr_lambda)
    for i in loop:
        indices = torch.randperm(pcd_points.size(0))  # 生成一个从 0 到 1023 的随机排列索引
        shuffled_pcd_points = pcd_points[indices]

        vertices = model(shuffled_pcd_points.repeat(batch_size, 1, 1), mv_points.repeat(batch_size, 1, 1)) # MLP:(B, N), review (B, -1, 3)
        patches = vertices[:,face_idx] # (B, face_num, cp_num, 3)
        curves = vertices[:,curve_idx] # (B, curve_num, cp_num, 3)
        mview_weights = 1 + (mview_weights * i / kwargs['epoch']) # 0.75~1.25
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
            'i':i, 
            'epoch_num':kwargs['epoch']
        }
        loss = compute_loss(**loss_params)
        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print(f"Epoch {i + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

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

                # save nn model
                torch.save(model, f"{output_path}/model_weights.pth")

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="output_cat_test")
    parser.add_argument('--mv_path', type=str, default="render_utils/train_outputs")

    parser.add_argument('--epoch', type=int, default="201")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.0005")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=3) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2) 

    args = parser.parse_args()
    training(**vars(args))