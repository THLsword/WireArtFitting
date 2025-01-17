import os
import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image
from sklearn.decomposition import PCA

from .utils.layers import Embedding_
from .backbone.apes_seg_backbone import APESSeg2Backbone
from .backbone.apes_cls_backbone import APESClsBackbone, simple_mlp, simple_mlp_
from .head.apes_cls_head import MLP_Head
from .utils.layers import UpSample


class Model(nn.Module):
    def __init__(self, template_params):
        super(Model, self).__init__()
        self.register_buffer('template_params', template_params) # (B, 122, 3)
        self.cp_num = self.template_params.shape[1] * self.template_params.shape[2]

        self.embedding = Embedding_()

        self.simple_MLP  = simple_mlp()
        self.simple_MLP2 = simple_mlp_() 
        self.upsample = UpSample()

        self.head = MLP_Head()

    def forward(self, pcd, mv_points):
        pcd = torch.transpose(pcd, 1, 2) # (B, N, 3) -> (B, 3, N)
        mv_points = torch.transpose(mv_points, 1, 2) # (B, M, 3) -> (B, 3, M)

        pcd_feature = self.embedding(pcd)
        pcd_feature = self.simple_MLP(pcd_feature) # (B, C, N)
        output = self.head(pcd_feature) # (B, C, N) -> (B, 366)

        # ---- 註釋掉就沒有cross attention
        ds_feature = self.embedding(mv_points)
        ds_feature = self.simple_MLP2(ds_feature) # (B, C, N)
        temp = self.upsample(pcd_feature, ds_feature) # (B, C, N)
        output = self.head(temp)
        # ---- 註釋掉就沒有cross attention
        
        output = rearrange(output, 'B (M N) -> B M N', N = 3)
        output = self.template_params + output

        return output

class WarmupModel(nn.Module):
    def __init__(self, template_params, batch_size):
        super(WarmupModel, self).__init__()
        self.batch_size = batch_size
        # template_params input shape : [B, N, C]
        self.register_buffer('template_params', template_params[0].reshape(-1)) # [N*C]

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_params)))

    def forward(self):
        vertices = self.template_params + self.displace # [N*C]
        return rearrange(vertices, '(N C) -> N C',C=3)