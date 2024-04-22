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
from utils.mview_utils import * 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(3, 8), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(8, 2))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(2, 8), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(8, 3))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Encoder_(nn.Module):
    def __init__(self):
        super(Encoder_, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(4096*3, 2048), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(2048, 1024), nn.Sigmoid())
        self.fc3 = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = x.view(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Decoder_(nn.Module):
    def __init__(self, init_params):
        super(Decoder_, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 1024), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(1024, 2048), nn.Sigmoid())
        self.fc3 = nn.Linear(2048, 4096*3)
        # nn.init.zeros_(self.fc3.weight)
        self.fc3.bias.data = init_params.clone()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(4096, -1)
        return x

class Model(nn.Module):
    def __init__(self, init_params):
        super(Model, self).__init__()
        self.encoder = Encoder_()
        self.decoder = Decoder_(init_params)

    def forward(self, pcd):
        uv = self.encoder(pcd)
        output = self.decoder(uv)

        return output, uv

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

    # train
    batch_size = kwargs['batch_size']
    control_point_num = 4
    model = Model(pcd_points.view(-1)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'], betas=(0.5, 0.99))
    loop = tqdm.tqdm(list(range(0, kwargs['epoch'])))

    for i in loop:
        vertices, uv = model(pcd_points)
        
        # chamfer loss
        vertices_ = vertices.unsqueeze(0)
        pcd_points_ = pcd_points.unsqueeze(0)
        chamferloss_a, idx_a, chamferloss_b, idx_b = curve_chamfer(vertices_, pcd_points_)
        chamfer_loss = (chamferloss_a.mean(1) + chamferloss_b.mean(1))/2

        # L2 loss
        mse_loss = F.mse_loss(vertices, pcd_points)*10

        # L1 loss
        L1_loss = (pcd_points - vertices).abs().mean()
        # print(vertices.shape)

        loss = L1_loss
        
        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == batch_size - 1:
            # save model
            output_path = kwargs['ae_output_path']
            os.makedirs(output_path, exist_ok=True)
            torch.save(model.state_dict(), f'{output_path}/model_parameters.pth')

            # save output pcd
            save_obj(f'{output_path}/output_pcd.obj', vertices)
            save_obj(f'{output_path}/gt.obj', pcd_points)

def testing(**kwargs):
    # load model
    batch_size = kwargs['batch_size']
    model = Model(batch_size).cuda()
    output_path = kwargs['output_path']
    model_parameters = torch.load(f'{output_path}/model_parameters.pth')
    model.load_state_dict(model_parameters)
    return 0

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    # parser.add_argument('--template_path', type=str, default="data/templates/animel122")
    parser.add_argument('--output_path', type=str, default="output_cat_24_m")
    parser.add_argument('--ae_output_path', type=str, default="output_ae")

    parser.add_argument('--epoch', type=int, default="1000")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.01")

    args = parser.parse_args()
    training(**vars(args))
    # testing(**vars(args))