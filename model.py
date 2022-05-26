from pyrsistent import T
import scipy.io as io
from glob import glob
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv2d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size, drop=0.3):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=1, groups=nin),
            nn.BatchNorm2d(nin * kernels_per_layer), 
            nn.ELU(), 
            nn.AvgPool2d((1, 8)), 
            nn.Dropout(drop)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=(1, 16), padding=0),
            nn.BatchNorm2d(nout), 
            nn.ELU(), 
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop)
        )
        

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DilatedCNN(nn.Module):
  def __init__(self, nin, kernel_size):
    super(DilatedCNN,self).__init__()
    self.convlayers = nn.Sequential(
      nn.Conv2d(in_channels = nin, out_channels = 6, kernel_size = kernel_size, stride = 1, padding = 0, dilation=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=6, out_channels=16, kernel_size = 3, stride = 1, padding= 0, dilation = 2),
      nn.ReLU(),
    )
    self.fclayers = nn.Sequential(
      nn.Linear(2304,120),
      nn.ReLU(),
      nn.Linear(120,84),
      nn.ReLU(),
      nn.Linear(84,10)
    )
  def forward(self,x):
    x = self.convlayers(x)
    x = x.view(-1,2304)
    x = self.fclayers(x)
    return x

class TempCNN(nn.Module):
    def __init__(self):
        super(TempCNN, self).__init__()
        self.F1 = 24
        self.Ke = 32
        self.C = 22
        self.D = 2
        self.F2 = 250   # ??
        self.T = 250
        self.L = 2
        self.Ft = 12
        self.Kt = 4
        self.pe = 0.3

        #block 1
        self.conv3x = nn.Sequential(
            TemporalBlock(n_inputs=250, n_outputs=self.F1, kernel_size=(1, self.Ke), stride=1, dilation=1, padding=0, dropout=self.pe), 
            nn.BatchNorm2d(self.F1), 
            nn.ELU(),
            depthwise_separable_conv(nin=self.F1, kernels_per_layer=self.D, nout=self.F2, kernel_size=(self.C, 1)),
            nn.AvgPool2d((1, 8)), 
            nn.Dropout(self.pe)
        )
        
        # block 2
        self.residual_block = nn.Sequential(
            DilatedCNN(self.Ft, self.Kt), 
            nn.BatchNorm2d(self.Kt), 
            nn.ELU(), 
            nn.Dropout(0.3),
            DilatedCNN(self.Ft, self.Kt), 
            nn.BatchNorm2d(self.Kt), 
            nn.ELU(), 
            nn.Dropout(0.3)
        )
        self.conv1x1 = nn.Conv2d(in_channels=self.Ft, out_channels=self.Ft, kernel_size=1)

        # block 3
        self.flatten = nn.Flatten()

    def forward(self, x):
        # do the 1st block
        x = self.conv3x(x)

        # do the 2nd block
        for _ in range(self.L):
            res_x = self.residual_block(x)
            one_x = self.conv1x1(x)
            x = one_x.concat(res_x)

        # do the 3rd block
        conc = x.concat(res_x, dim=0)
        out = self.flatten(x).stack(self.flatten(conc), dim=0)
        return out


DATA_PATH = glob('./data/*.mat')
data = [io.loadmat(path) for path in DATA_PATH]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TempCNN()