"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import math
import numpy as np
import argparse
from einops import rearrange, repeat
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from PIL import Image

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

class Model(nn.Module):
    def __init__(self, pcd, device):
        super(Model, self).__init__()
        self.device = device
        # set template mesh
        self.pcd = pcd.to(device) # [2048,3]
        self.register_buffer('init_colors', torch.zeros(4096))

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.init_colors)))

        # render
        self.views = [45,90,135,225,270,315] # degree of 360
        self.view_num = len(self.views)
        self.R, self.T = look_at_view_transform(1.5, 15, self.views) 
        self.raster_settings = PointsRasterizationSettings(
            image_size=128, 
            radius = 0.01,
            points_per_pixel = 5
        )
        self.cameras = FoVOrthographicCameras(device=device, R=self.R, T=self.T, znear=0.01)
        self.rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )

    def forward(self):
        base = self.init_colors.to(self.device)
        colors_ = torch.sigmoid(base+self.displace) # [2048]

        points = self.pcd.unsqueeze(0).repeat(self.view_num,1,1)
        colors = colors_.unsqueeze(1).repeat(1,3) # (2048) -> (2048,3)
        colors = colors.unsqueeze(0).repeat(self.view_num,1,1)

        # point_cloud = Pointclouds(points=[self.pcd], features=[colors])
        point_cloud = Pointclouds(points=[points[i] for i in range(points.shape[0])], 
                                features=[colors[i] for i in range(colors.shape[0])])
        images = self.renderer(point_cloud) # (self.view_num,256,256,3)
        return images, colors_

def WeightL1(pred, target):
    pred = pred.sum(dim=-1)/3     # (B,4,256,256,3) -> (B,4,256,256)
    target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256)

    # count_all = torch.sum((pred <= target) & (target > 0))
    count_all = torch.sum(target > 0)
    L1_loss = torch.where(pred <= target, (pred - target)*2, (pred - target)*0.5)
    # L1_loss = pred - target
    L1_loss = L1_loss.abs()
    L1_loss = L1_loss.sum()/count_all

    return L1_loss

def main(DATA_DIR, pcd_path, gt_path, output_path, epoch):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # load pcd
    npzfile = np.load(pcd_path)
    pcd = npzfile['points']
    # 将点云平移到原点
    centroid = np.mean(pcd, axis=0)
    point_cloud_centered = pcd - centroid
    # 缩放点云使其适应[-1, 1]范围
    max_distance = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
    point_cloud_normalized = point_cloud_centered / max_distance
    pcd_tensor = torch.tensor(point_cloud_normalized).to(torch.float32).to(device)

    # load gt(expened alpha shape images)
    multi_view_paths = []
    for filename in os.listdir(gt_path):
        if filename.endswith('.png'):
            file_path = os.path.join(gt_path, filename)
            multi_view_paths.append(file_path)
    multi_view_paths.sort()

    multi_view_imgs = [Image.open(path) for path in multi_view_paths]
    np_imgs = [np.array(img) for img in multi_view_imgs]
    gt_np = [img.astype(np.float32) / 255. for img in np_imgs]
    images_gt = torch.tensor(gt_np).to(device)

    # init
    model = Model(pcd_tensor, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.1, betas=(0.5, 0.99))
    loss_fn = nn.MSELoss(reduction="mean")

    loop = tqdm.tqdm(list(range(0, epoch)))
    for i in loop:
        images, colors = model.forward()
        loss = WeightL1(images, images_gt)

        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            for j, image in enumerate(images):
                save_img(image.detach().cpu().numpy(), f'{output_path}/output_{i+1}_{j}.png')
            torch.save(colors, f'{DATA_DIR}/weights.pt')
            torch.save(colors, f'{output_path}/weights.pt')

        if (i+1) == epoch:
            mask = (colors > 0.5)
            masked_pcd = pcd_tensor[mask]
            save_obj(f'{output_path}/multi_view.obj', masked_pcd)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./data/models/cat")
    parser.add_argument('--GT_DIR', type=str, default="./render_utils/expand_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/train_outputs")
    parser.add_argument('--filename', type=str, default="model_normalized_4096.npz")
    parser.add_argument('--epoch', type=int, default=50)

    args = parser.parse_args()

    file_path = os.path.join(args.DATA_DIR, args.filename)
    # Set paths
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR, exist_ok=True)

    main(args.DATA_DIR, file_path, args.GT_DIR, args.SAVE_DIR, args.epoch)