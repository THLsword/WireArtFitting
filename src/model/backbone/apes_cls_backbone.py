import torch
from torch import nn
from einops import pack
import os
import sys
# from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample, DownSample_new, GlobalDownSample_more

class APESClsBackbone(nn.Module):
    def __init__(self, which_ds):
        super(APESClsBackbone, self).__init__()
        self.embedding = Embedding()
        if which_ds == 'global':
            self.ds1 = GlobalDownSample_more(2048)  # N -> 2048 pts
            self.ds2 = GlobalDownSample_more(1024)  # N -> 1024 pts
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = LocalDownSample(512)  # 1024 pts -> 512 pts
        elif which_ds == 'new':
            self.ds1 = DownSample_new(1024)  # 2048 pts -> 1024 pts
            self.ds2 = DownSample_new(512)  # 1024 pts -> 512 pts
        else:
            raise NotImplementedError
        self.n2p_attention = N2PAttention()
        self.n2p_attention1 = N2PAttention()
        self.n2p_attention2 = N2PAttention()
        self.conv = nn.Conv1d(128, 1024, 1)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
         
        self.conv3 = nn.Sequential(nn.Conv1d(384, 512, 1, bias=False), nn.LayerNorm(512), nn.Sigmoid())
        self.conv4 = nn.Sequential(nn.Conv1d(512, 128, 1, bias=False), nn.LayerNorm(128), nn.Sigmoid())

        self.conv_simple1 = nn.Sequential(nn.Conv1d(3, 32, 1, bias=False), nn.LayerNorm(32), nn.LeakyReLU(0.2))
        self.conv_simple2 = nn.Sequential(nn.Conv1d(32, 128, 1, bias=False), nn.LayerNorm(128), nn.LeakyReLU(0.2))

    def forward(self, x):
        pcd_num = x.shape[2]
        self.res_link_list = []
        x = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        x = self.n2p_attention(x)  # (B, 128, 2048) -> (B, 128, 2048)
        self.res_link_list.append(self.conv(x).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        x, _ = self.ds1(x)  # (B, 128, 2048) -> (B, 128, 1024)
        x = self.n2p_attention1(x)  # (B, 128, 1024) -> (B, 128, 1024)
        self.res_link_list.append(self.conv1(x).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)
        x, _ = self.ds2(x)  # (B, 128, 1024) -> (B, 128, 512)
        x = self.n2p_attention2(x)  # (B, 128, 512) -> (B, 128, 512)
        self.res_link_list.append(self.conv2(x).max(dim=-1)[0])  # (B, 128, 512) -> (B, 1024, 512) -> (B, 1024)
        x, ps = pack(self.res_link_list, 'B *')  # (B, 3072)
        return x 
 
class simple_mlp(nn.Module):
    def __init__(self):
        super(simple_mlp, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3,128,1), nn.LeakyReLU(0.2))
        self.n2p_attention = N2PAttention()

    def forward(self, x):
        x = self.conv1(x)
        x = self.n2p_attention(x)
        # x, _ = x.max(dim = 2)

        return x
