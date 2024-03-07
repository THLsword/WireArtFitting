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

from dataset.load_npz import load_npz
from dataset.load_template import load_template

class Model(nn.Module):
    def __init__(self, template_params, batch_size):
        super(Model, self).__init__()
        self.template_params = template_params
        self.batch_size = batch_size
        self.register_buffer('template_params', template_params)

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_params)))

    def forward(self):
        vertices = self.template_params + self.displace

        return vertices
        


def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)


def main():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # load .npz point cloud
    model_path = "data/models/cat.npz"
    pcd_points, pcd_normals = load_npz(model_path)

    # load template
    template_path = "data/templates/sphere24"
    template_params, patch_kwargs, curve_kwargs = load_template(template_path)

    # train
    model = Model(template_params, 8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    loop = tqdm.tqdm(list(range(0, 2000)))

    for i in loop:
        vertices = model()

if __name__ == '__main__':
    # ` python src/deform.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename-input', type=str, default="111")

    args = parser.parse_args()
    main()