import os, math, copy
import itertools
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
from torch.nn import init
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0) # shape(1, seq_len, 30)
        self.pe = pe
 
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        bs, seq_len = x.size(0), x.size(1)
        pe = self.pe[:, :seq_len, :]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe_all = Variable(torch.zeros(bs, seq_len, self.d_model), requires_grad=True).to(device)
        for i in range(bs):
            pe_all[i, :, :] = pe

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1,
                activation="relu", normalize_before=False):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.d_model = d_model
        self.pe = PositionalEncoder(d_model)

    def forward_post(self,
                     src,
                     src_mask= None,
                     src_key_padding_mask= None,
                     pos= None):
        q = k = self.pe(src)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask= None,
                    src_key_padding_mask= None,
                    pos= None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask= None,
                src_key_padding_mask= None,
                pos= None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask= None,
                src_key_padding_mask= None,
                pos= None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer(nn.Module):
    def __init__(self, d_model=30, nhead=3, num_encoder_layers=6,
                 dim_feedforward=128, dropout=0.1,
                 num_joints_in=15, num_joints_out=15,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.linear = nn.Linear(num_joints_in*2, num_joints_out*3, bias=False)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_out = num_joints_out
        self.pe = PositionalEncoder(d_model)

    def forward(self, src, mask=None, query_embed=None, pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs = src.shape[0]
        src = torch.flatten(src, start_dim=2)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output = self.linear(memory)
        ran = [13, output.size(1)-13]
        output = output[:, ran[0]:ran[1], :] # [b,n-26,45]
        # print(output.shape)

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return output.reshape(bs, -1, self.num_joints_out, 3)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    model = Transformer()
    x = torch.rand(128, 27, 15, 2)
    output = model(x)
    print(output.shape)
