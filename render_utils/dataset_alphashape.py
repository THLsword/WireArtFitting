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

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def extract_points(image):
    # 将图片转换为灰度
    gray = np.mean(image, axis=-1)
    
    # 找到所有不是黑色的点的坐标
    y, x = np.where(gray > 0)
    
    # 调整y坐标
    y = image.shape[0] - 1 - y
    
    # 组合x和y坐标
    points_2d = list(zip(x, y))
    
    return points_2d

def alpha(DATA_DIR,SAVE_DIR,filename):
    # 读取 PNG 图片
    file_path = os.path.join(DATA_DIR, filename)
    image = Image.open(file_path)

    # 将图片转换为 numpy 数组
    data = np.array(image)
    points_2d = extract_points(data)

    shaper = Alpha_Shaper(points_2d)
    try:
        alpha = 30.0
        alpha_shape = shaper.get_shape(alpha=alpha)
        
        vertices = []
        # print(filename)
        for boundary in get_boundaries(alpha_shape):
            exterior = Path(boundary.exterior)
            holes = [Path(hole) for hole in boundary.holes]
            path = Path.make_compound_path(exterior, *holes)
            vertices.append(path.vertices)
    except TypeError:
        alpha = 3.0
        alpha_shape = shaper.get_shape(alpha=alpha)
        
        vertices = []
        print("TypeError: ",filename)
        for boundary in get_boundaries(alpha_shape):
            exterior = Path(boundary.exterior)
            holes = [Path(hole) for hole in boundary.holes]
            path = Path.make_compound_path(exterior, *holes)
            vertices.append(path.vertices)

    npvertices = np.concatenate(vertices)

    img = Image.new('RGB', (256, 256), color='black')
    pixels = img.load()

    # 将数组中的每个点设置为白色
    for point in npvertices:
        x, y = point
        neighbors = [(x + dx, y + dy) for dy in range(-1, 2) for dx in range(-1, 2)]
        for nx, ny in neighbors:
            nx=int(nx)
            ny=int(ny)
            # Check boundaries
            if 0 <= nx < data.shape[0] and 0 <= ny < data.shape[1]:
                # print(data[nx, ny])
                # pixels[nx, data.shape[0]-1-ny] = (0, 0, 0)

                if np.any(data[data.shape[0]-1-ny, nx] > 0):
                    pixels[nx, data.shape[0]-1-ny] = (255, 255, 255)

    SAVE_filename=f'{os.path.splitext(filename)[0]}.png'
    img.save(os.path.join(SAVE_DIR,SAVE_filename))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/alpha_outputs")

    args = parser.parse_args()

    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    for filename in tqdm(os.listdir(args.DATA_DIR)):
        if filename.endswith('.png'):
            file_path = os.path.join(args.DATA_DIR, filename)
            alpha(args.DATA_DIR,args.SAVE_DIR,filename)
