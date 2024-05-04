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


def curvature_loss(curves, linspace):
    # Reduce curvature and make curves straighter
    # print(curves.shape) # [1, 96, 4, 3]
    # print(linspace.shape) # [16, 1]

    curvature_lists = []
    t = linspace.unsqueeze(1).repeat(1,3) # (16,3)
    for i in curves[0]:
        p0 = i[0]
        p1 = i[1]
        p2 = i[2]
        p3 = i[3]
        curvature_list = []

        B = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        # First derivative
        B_prime = 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
        # Second derivative
        B_double_prime = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
        # curvature
        # print(B_prime)
        # print(B_double_prime)
        cross_product = torch.cross(B_prime, B_double_prime)
        numerator = torch.norm(cross_product, dim=1)
        # print(numerator)
        denominator =torch.norm(B_prime, dim=1)**3
        curvature_list.append(numerator / denominator)
        # print(curvature_list)

    #     curvature_lists.append(curvature_list)
    # curvature = torch.tensor(curvature_lists)
    # print(curvature)


t_values = torch.linspace(0, 1, 10)
curves = torch.tensor([[0,0,0],[0,0.8,1],[0,2.2,1], [0,3,0]])
curves = curves.unsqueeze(0).unsqueeze(0)
curvature_loss(curves, t_values)
weights = [(i-0.5)**2 for i in t_values]
# print(weights)


t = (1-t_values)**3
t = t.unsqueeze(1).repeat(1,3) # (10,3)



a = torch.ones(10,3)
b = torch.tensor([1,2,3])
b = b.repeat(96,1,1)
print(b.shape)
c = a*b
print(c.shape)