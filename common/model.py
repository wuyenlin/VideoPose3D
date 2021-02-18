# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, math
import itertools
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)
        
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        
        return x    

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x
    
class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x
 
class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=3, num_layers=3):
        """
        MLP or FFN mentioned in the paper
        (3-layer perceptron with 
            1. ReLU activation function
            2. hidden dimension d
            3. linear projection layer
        )
        In the end, FFN predicts the box coordinates (normalized), width and height;
                linear layer + Softmax predicts LABEL.

        Note: 
        input_dim = hidden_dim = transformer.d_model
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n,k)
                                    for n,k in zip([input_dim] + h, h + [output_dim]) )
        # same as creating 3 linear layers: 64,64; 64,64; 64,3
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x   


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
        pe = pe.unsqueeze(0) # shape(1, seq_len, 34)
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
        # return self.dropout(x)    
        return x

class NewPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0) # shape(1, seq_len, 34)
        self.pe = pe
 
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        bs, d_model,seq_len = x.size(0), x.size(1), x.size(2)
        pe = self.pe[:, :d_model, :seq_len]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe_all = Variable(torch.zeros(bs, d_model, seq_len), requires_grad=True).to(device)
        for i in range(bs):
            pe_all[i, :, :] = pe

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return self.dropout(x)  

class tppe(nn.Module):
    """
    Implementation of 2D positional encoding used in TransPose
    """
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1, bs=128):
        super().__init__()
        self.bs = bs
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pe_x = Variable(torch.zeros(bs, max_seq_len, d_model), requires_grad=False).to(device)
        self.pe_y = Variable(torch.zeros(bs, max_seq_len, d_model), requires_grad=False).to(device)

        for i in range(0, bs, 2):
            for combi in itertools.zip_longest(range(d_model), range(max_seq_len)):
                if combi[0] is not None:
                    p_x = combi[0]
                    self.pe_x[i, :, p_x] = math.sin(2*math.pi*p_x/ (d_model* 10000 ** ((2 * i)/d_model)))
                    self.pe_x[i+1, :, p_x] = math.cos(2*math.pi*p_x/ (d_model* 10000 ** ((2 * (i+1))/d_model)))

                if combi[1] is not None:
                    p_y = combi[1]
                    self.pe_y[i, p_y, :] = math.sin(2*math.pi*p_y/ (max_seq_len * 10000 ** ((2 * i)/d_model)))
                    self.pe_y[i+1, p_y, :] = math.cos(2*math.pi*p_y/ (max_seq_len * 10000 ** ((2 * (i+1))/d_model)))
 

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        bs, seq_len = x.size(0), x.size(1)
        pe_x = self.pe_x[:bs, :seq_len, :]
        pe_y = self.pe_y[:bs, :seq_len, :]

        assert x.shape == pe_x.shape, "{},{}".format(x.shape, pe_x.shape)
        assert x.shape == pe_y.shape, "{},{}".format(x.shape, pe_y.shape)
        x = x + pe_x + pe_y
        return self.dropout(x) 

class firstTransformer(nn.Module):
    def __init__(self, d_model=34, nhead=2, num_layers=6, 
                    num_joints_in=15, num_joints_out=15):
        super().__init__()
        # self.pe = tppe(d_model=d_model)
        self.pe = PositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(num_joints_in*2, num_joints_out*3, bias=False)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, x, pe=False):
        sz = x.shape[:3]
        x = torch.flatten(x, start_dim=2)
        if pe:
            x = self.pe(x) 
        x = self.transformer(x) 
        x = self.linear(x) 
        ran = [13, x.size(1)-13]
        x = x[:, ran[0]:ran[1], :] # [b,n-26,45]

        return x.reshape(sz[0], -1, self.num_joints_out, 3)

class LiftFormer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, 
                    num_joints_in=17, num_joints_out=15):
        super().__init__()
        self.conv_in = nn.Conv1d(num_joints_in*2, d_model, kernel_size=3, padding=1, bias=False)
        self.pe = NewPositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.conv_out = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, bias=False),
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=3, bias=False),
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=9, bias=False),
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model, num_joints_out*3, kernel_size=1, bias=False),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, x):
        sz = x.shape[:3]
        x = torch.flatten(x, start_dim=2) # from [b, n, 17, 2] to [b, n, 34]
        x = x.permute(0, 2, 1) # [b, 34, n]
        x = self.conv_in(x) # [b, 512, n]
        x = x.permute(0, 2, 1) 
        x = self.transformer(self.pe(x)) 
        x = x.permute(0, 2, 1) 
        x = self.conv_out(x) # [b, 45, 1]
        x = x.permute(0, 2, 1) # [b, 1, 45]

        return x.reshape(sz[0], -1, self.num_joints_out, 3)
        
