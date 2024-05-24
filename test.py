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

a = torch.tensor([1,2,3,4])

mask1 = a == 1 
mask2 = a == 2
mask = not (mask1 & mask2)
print(mask1)
print(mask2)
print(mask)