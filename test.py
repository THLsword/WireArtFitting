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
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline

# a = torch.randint(1,5,[1,4,3])
# print(a)

# b = a.repeat_interleave(2, dim=0)
# print(b)
# print(b.shape)

# c = a.repeat(2,1,1)
# print(c)

a = torch.randint(1,5,[4,3])

a = a.repeat(1,1,3)
print(a.shape)