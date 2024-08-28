import torch
import torch.nn as nn
import torch.nn.functional as F
#model augmentations:
#window type
#stereo
#mixup

class WHEncLayer(nn.Module):
    def __init__(self, in_c, out_c):
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=8, stride=4)
        self.BN = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
    def forward(self,x, skip=True):
        c = self.conv(x)
        x = self.BN(c)
        x = self.act(x)
        if skip:
            return x, c
        return x

class WHDecLayer(nn.Module):
    def __init__(self, in_c, out_c):
        self.conv = nn.ConvTranspose1d(in_c, out_c, kernel_size=8, stride=4)
        self.BN = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
    def forward(self,x, skip=None):
        if skip is not None:
            x += skip
        x = self.conv(x)
        x = self.BN(x)
        x = self.act(x)
        return x

class MagicalMidLayer(nn.Module):
    def __init__(self, rnn_in, rnn_hidden):
        self.contextRNN = nn.RNN(rnn_in, rnn_hidden)
        #transf encoder layer for context and current
        #cross attention
        #repeat twice
        #reshape and output current
    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.act(x)
        return x


class WHAMSAHNet(nn.Module):
    def __init__(self, cfg):
        super(WHAMSAHNet, self).__init__()
        self.depth = cfg['model']['depth']
        self.growth = cfg['model']['growth']
        self.in_c = 2
        self.out_c = 4 # 2 speakers * 2 parts
        self.ENCLAYERS = nn.ModuleList()
        self.DECLAYERS = nn.ModuleList()
        for i in range(self.depth):
            self.ENCLAYERS.append(WHEncLayer(self.in_c, self.in_c*self.growth))
            self.in_c = self.in_c*self.growth
        for i in range(self.depth):
            self.DECLAYERS.append(WHDecLayer(self.in_c, self.in_c//self.growth))
            self.in_c = self.in_c//self.growth
    
    def forward(self, x):
        return x