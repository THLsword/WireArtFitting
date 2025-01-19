import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from sklearn.decomposition import PCA

from .utils.layers import UpSample
from .backbone.apes_cls_backbone import PcdBackbone, PrepBackbone
from .head.apes_cls_head import MLPHead



class Model(nn.Module):
    def __init__(self, template_params):
        super(Model, self).__init__()
        self.register_buffer('template_params', template_params) # (B, 122, 3)
        self.cp_num = self.template_params.shape[1] * self.template_params.shape[2]

        self.pcd_backbone  = PcdBackbone()
        self.prep_backbone = PrepBackbone() 
        self.upsample = UpSample()

        self.head = MLPHead()

    def forward(self, pcd, prep_points):
        pcd = torch.transpose(pcd, 1, 2) # (B, N, 3) -> (B, 3, N)
        prep_points = torch.transpose(prep_points, 1, 2) # (B, M, 3) -> (B, 3, M)

        pcd_feature = self.pcd_backbone(pcd) # (B, C, N)
        # output = self.head(pcd_feature) # (B, C, N) -> (B, 366)

        # ---- 註釋掉就沒有cross attention
        ds_feature = self.prep_backbone(prep_points) # (B, C, N)
        temp = self.upsample(pcd_feature, ds_feature) # (B, C, N)
        output = self.head(temp)
        # ---- 註釋掉就沒有cross attention
        
        output = rearrange(output, 'B (M N) -> B M N', N = 3)
        output = self.template_params + output

        return output
