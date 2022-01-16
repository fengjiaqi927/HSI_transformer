import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer 

# vit based model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1 :
            pe[:, 1::2] = torch.cos(position * div_term)[:,:len(div_term)-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(2),0, :] 
        return self.dropout(x)


class GaussEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=300):
        super(GaussEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1 :
            pe[:, 1::2] = torch.cos(position * div_term)[:,:len(div_term)-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        gauss_kernel = self.gauss(int(np.sqrt(d_model)),int(np.sqrt(d_model)/2)).reshape((-1,d_model))
        self.register_buffer('gauss_kernel', torch.from_numpy(gauss_kernel).to(torch.float32))

    
    def gauss(self,kernel_size, sigma):
        
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = kernel_size // 2
       
        s = sigma ** 2
        sum = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * s))
                sum += kernel[i, j]
        
        return kernel

    def forward(self, x):
        x = x + self.pe[:x.size(2),0, :]*self.gauss_kernel
        return self.dropout(x)

class spectral_attention(nn.Module):
    def __init__(self,
            patch_size, # HSI channel merge size 
            sample_size, # sample area size = h*w (default shape : square  ) 
            head_num=16, # head_num in transformer backbone
            num_classes = 16,
            pool_method = 'mean'
            ):
        super().__init__()
        self.pool = pool_method
        self.patch_size = patch_size
        # merged patch feature dim, default = sample_size * patch_size
        self.patch_feature_dim = patch_size * sample_size

        # self.cls_token = nn.Parameter(torch.randn(1, 1, sample_size))


        # position_enc 1 :
        # self.position_enc = PositionalEncoding(d_model = sample_size)
        self.position_enc = GaussEncoding(d_model = sample_size)

        # position_enc 2 :
        # temp_size = int(math.sqrt(sample_size))
        # self.position_enc = nn.Parameter(torch.randn(1,1,1,temp_size, temp_size))
        # self.position_enc = nn.Parameter(torch.zeros(1,1,1,temp_size, temp_size))
        # trunc_normal_(self.position_enc, std=.02)
        # self.position_enc[0,0,0,2,2] = 1 
        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b no_sense (c_num c_patch) h w -> b no_sense c_num (c_patch h w)',c_patch = patch_size),
            nn.Linear(sample_size * patch_size, self.patch_feature_dim),
        )

        self.transformer = Transformer(dim=self.patch_feature_dim, depth=1, heads=head_num,
            dim_head=self.patch_feature_dim//(head_num//4), mlp_dim = self.patch_feature_dim*2, dropout=0.1)

        # self.to_origin_embedding = nn.Sequential(
        #     Rearrange('b no_sense c_num (c_patch h w) -> b no_sense (c_num c_patch) h w',c_patch = patch_size),
        #     nn.Linear(self.patch_feature_dim, self.patch_feature_dim),
        # )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.patch_feature_dim),
            nn.Linear(self.patch_feature_dim, num_classes)
        )

        # self.batch_norm = nn.BatchNorm1d(self.patch_feature_dim)
        



    def forward(self,x):
        x = x.unsqueeze(1)

        # step1 : channels padding 
        C = x.shape[2]
        add_channel_nums = self.patch_size - C % self.patch_size
        x = F.pad(x, pad=(0, 0, 0, 0, 0, add_channel_nums), mode="replicate")

        # step2 : position_embedding
        b,_,c,h,w = x.shape
        x = x.view(b,-1,c,h*w)
        x = self.position_enc(x)
        x = x.view(b,-1,c,h,w)
        # x = self.position_enc + x

        # step3 : MSA
        # input x.shape  batch * 1 * channel * height * width
        x = self.to_patch_embedding(x)
        x = self.transformer(x.squeeze(dim=1))

        # step4 :  
        # x = self.to_origin_embedding(x)
        # x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

