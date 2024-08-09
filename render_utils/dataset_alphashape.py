import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# Util function for loading point clouds|
import numpy as np

from PIL import Image

from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm

def extract_points(image):
    gray = np.mean(image, axis=-1)
    y, x = np.where(gray > 0)
    y = image.shape[0] - 1 - y
    points_2d = list(zip(x, y))
    
    return points_2d

def alpha(DATA_DIR,SAVE_DIR,filename, alpha_size):
    # 读取 PNG 图片
    file_path = os.path.join(DATA_DIR, filename)
    image = Image.open(file_path)

    # 将图片转换为 numpy 数组
    data = np.array(image)
    points_2d = extract_points(data)

    # alpha shape
    shaper = Alpha_Shaper(points_2d)
    try:
        alpha = alpha_size
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

    img = Image.new('RGB', (128, 128), color='black')
    pixels = img.load()

    # 将数组中的每个点设置为白色
    for point in npvertices:
        x, y = point
        neighbors = [(x + dx, y + dy) for dy in range(-1, 2) for dx in range(-1, 2)]
        for nx, ny in neighbors:
            nx=int(nx)
            ny=int(ny)
            # Check boundaries,確保不會將render img中黑色的背景塗白
            if 0 <= nx < data.shape[0] and 0 <= ny < data.shape[1]:
                if np.any(data[data.shape[0]-1-ny, nx] > 0):
                    pixels[nx, data.shape[0]-1-ny] = (255, 255, 255)

    SAVE_filename=f'{os.path.splitext(filename)[0]}.png'
    img.save(os.path.join(SAVE_DIR,SAVE_filename))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--alpha_size', type=float, default=50.0)

    args = parser.parse_args()

    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    for filename in tqdm(os.listdir(args.DATA_DIR)):
        if filename.endswith('.png'):
            file_path = os.path.join(args.DATA_DIR, filename)
            alpha(args.DATA_DIR, args.SAVE_DIR, filename, args.alpha_size)
