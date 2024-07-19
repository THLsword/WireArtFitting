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
import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.graph['1']='1'
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)
G.add_edge(2, 3)

print(G.nodes)
print(G.edges)

print(G.adj[1])
print(G.graph)