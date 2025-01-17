import os
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image
from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm
import time


def extract_points(image):
    gray = np.mean(image, axis=-1)
    y, x = np.where(gray > 0)
    y = image.shape[0] - 1 - y
    points_2d = list(zip(x, y))
    
    return points_2d

def img_alphashape(input_img, alpha_size, expand_size: int):
    assert expand_size >= 1, f"expand_size must be >= 1, but got {expand_size}"

    # 将图片转换为 numpy 数组
    data = np.array(input_img)
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

    # 创建黑色背景图像
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    # 遍历点并设置为白色
    for point in npvertices:
        x, y = point
        neighbors = [(x + dx, y + dy) for dy in range(-expand_size, expand_size + 1) for dx in range(-expand_size, expand_size + 1)]
        for nx, ny in neighbors:
            nx = int(nx)
            ny = int(ny)
            # Check boundaries, 确保不将背景设置为白色
            if 0 <= nx < data.shape[0] and 0 <= ny < data.shape[1]:
                if np.any(data[data.shape[0]-1-ny, nx] > 0):
                    img[data.shape[0]-1-ny, nx] = [255, 255, 255]


    return img

def multi_process_image(params):
    """
    多进程调用时的处理函数。  
    params: (image, alpha_size, expand_size)  
    將參數傳給img_alphashape(input_img, alpha_size, expand_size: int)
    """
    image, alpha_size, expand_size = params
    contour_img = img_alphashape(image, alpha_size, expand_size)
    return contour_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--ALPHA_SIZE', type=float, default=50.0)
    parser.add_argument('--EXPAND_SIZE', type=int, default=1)

    args = parser.parse_args()

    # create folder
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    start_time = time.time()

    images = []

    # ----- multi process
    start_time = time.time()
    loaded_images = []
    for filename in tqdm(os.listdir(args.DATA_DIR)):
        if filename.endswith('.png'):
            file_path = os.path.join(args.DATA_DIR, filename)
            image = Image.open(file_path)
            loaded_images.append(image)

    process_num = len(loaded_images) if len(loaded_images) <= 4 else 4
    with Pool(processes=process_num) as pool:
        params_list = [(img, args.ALPHA_SIZE, args.EXPAND_SIZE) for img in loaded_images]
        images = list(
            tqdm(
                pool.imap(multi_process_image, params_list),
                total=len(params_list),
                desc="Processing"
            )
        )
    end_time = time.time()
    print('timr: ', end_time - start_time)
    # -----

    # ----- single process
    start_time = time.time()
    for filename in tqdm(os.listdir(args.DATA_DIR)):
        if filename.endswith('.png'):
            file_path = os.path.join(args.DATA_DIR, filename)
            image = Image.open(file_path)
            contour_img = img_alphashape(image, args.ALPHA_SIZE, args.EXPAND_SIZE)
            images.append(contour_img)
            contour_img = Image.fromarray(contour_img)
            SAVE_filename=f'{os.path.splitext(filename)[0]}.png'
            contour_img.save(os.path.join(args.SAVE_DIR,SAVE_filename))
    end_time = time.time()
    print('timr: ', end_time - start_time)