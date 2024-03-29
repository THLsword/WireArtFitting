import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# Util function for loading point clouds|
import numpy as np

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

from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm

def extract_points(image):
    # 将图片转换为灰度
    gray = np.mean(image, axis=-1)
    
    # 找到所有不是黑色的点的坐标
    y, x = np.where(gray > 0)
    
    # 调整y坐标
    # y = image.shape[0] - 1 - y
    
    # 组合x和y坐标
    points_2d = list(zip(x, y))
    
    return points_2d

def Expand(alphashape_DIR, render_DIR, SAVE_DIR,filename, expand_size):
    # 读取 alphashape 图片
    file_path = os.path.join(alphashape_DIR, filename)
    image = Image.open(file_path)
    alphashape = np.array(image)
    alphashape_v = extract_points(alphashape)

    # 讀取 render 圖片
    file_path = os.path.join(render_DIR, filename)
    image2 = Image.open(file_path)
    render_img = np.array(image2)
    render_v = extract_points(render_img)

    img = Image.new('RGB', (256, 256), color='black')
    pixels = img.load()

    # 將新圖'pixels'中和'render'對應點的顏色變成'render'的X倍 OR 全白 OR 全黑
    scale = 0.0
    for point in render_v:
        x, y = point
        if np.any(render_img[y, x] > 0):
            rgb = render_img[y, x] * scale
            pixels[x, y] = tuple(int(value) for value in rgb)
    # 將新圖'pixels'中和'alphashape'對應點的顏色變成'render'的原本顏色 OR 全白
    for point in alphashape_v:
        x, y = point
        neighbors = [(x + dx, y + dy) for dy in range(-(expand_size - 1), expand_size) for dx in range(-(expand_size - 1), expand_size)]
        for nx, ny in neighbors:
            nx=int(nx)
            ny=int(ny)
            # Check boundaries
            if 0 <= nx < render_img.shape[0] and 0 <= ny < render_img.shape[1]:
                # print(render_img[nx, ny])
                # pixels[nx, render_img.shape[0]-1-ny] = (0, 0, 0)
                if np.any(render_img[ny, nx] > 0):
                    pixels[nx, ny] = tuple(render_img[ny, nx])
                    # pixels[nx, ny] = (255,255,255)

    SAVE_filename=f'{os.path.splitext(filename)[0]}.png'
    img.save(os.path.join(SAVE_DIR,SAVE_filename))
    

if __name__ == "__main__":
    # Set paths

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--alphashape_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/expand_outputs")
    parser.add_argument('--expend_size', type=int, default=4)

    args = parser.parse_args()

    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    for filename in tqdm(os.listdir(args.render_DIR)):
        if filename.endswith('.png'):
            Expand(args.alphashape_DIR,args.render_DIR,args.SAVE_DIR,filename,args.expend_size)
