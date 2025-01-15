import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_points(image):
    gray = np.mean(image, axis=-1)
    y, x = np.where(gray > 0)
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

    img = Image.new('RGB', (128, 128), color='black')
    pixels = img.load()

    # 將新圖'pixels'中和'alphashape'對應點的顏色變成'render'的原本顏色 OR 全白
    for point in alphashape_v:
        x, y = point
        neighbors = [(x + dx, y + dy) for dy in range(-(expand_size - 1), expand_size) for dx in range(-(expand_size - 1), expand_size)]
        for nx, ny in neighbors:
            nx=int(nx)
            ny=int(ny)
            # Check boundaries,確保不會將render img中黑色的背景塗白
            if 0 <= nx < render_img.shape[0] and 0 <= ny < render_img.shape[1]:
                if np.any(render_img[ny, nx] > 0):
                    pixels[nx, ny] = tuple(render_img[ny, nx])
 
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
