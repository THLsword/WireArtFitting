import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from sklearn.decomposition import PCA

from .utils.layers import UpSample, PcdEmbedding
from .backbone.backbone import PcdBackbone, PrepBackbone
from .head.head import MLPHead



class Model(nn.Module):
    def __init__(self, template_params):
        super(Model, self).__init__()
        self.register_buffer('template_params', template_params) # (B, 122, 3)
        self.cp_num = self.template_params.shape[1] * self.template_params.shape[2]
        self.pcd_backbone  = PcdBackbone()
        self.prep_backbone = PcdBackbone()
        self.upsample = UpSample()
        self.output_size = template_params.shape[1]*3
        self.head = MLPHead(self.output_size)
        self.embedding_layer = PcdEmbedding(8)

    def forward(self, pcd, prep_points):
        pcd = torch.transpose(pcd, 1, 2) # (B, N, 3) -> (B, 3, N)
        prep_points = torch.transpose(prep_points, 1, 2) # (B, M, 3) -> (B, 3, M)
        # feature extraction
            # point cloud
        pcd = self.embedding_layer(pcd)
        pcd_feature = self.pcd_backbone(pcd) # (B, C, N)
            # preprocessed point cloud
        prep_points = self.embedding_layer(prep_points)
        ds_feature = self.prep_backbone(prep_points) # (B, C, N)
        # cross attention 
        temp = self.upsample(pcd_feature, ds_feature) # (B, C, N)
        # head & output 
        output = self.head(temp)
        output = rearrange(output, 'B (M N) -> B M N', N = 3)
        output = self.template_params + output

        return output
