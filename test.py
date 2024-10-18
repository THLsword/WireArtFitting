import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.utils.postprocess_utils import render
from PIL import Image
import numpy as np
import alphashape
import threading
import psutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import networkx as nx

a = torch.tensor([[1,2,3],[1,2,3],[1,2,4]])
a_nui = torch.max(a, dim=0)

print(a)
print(a_nui)

b = np.array([])
print(b)