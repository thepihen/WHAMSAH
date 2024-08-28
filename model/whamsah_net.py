import torch
import torch.nn as nn
import torch.nn.functional as F
#model augmentations:
#window type
#stereo
#mixup

class WHAMSAHNet(nn.Module):
    def __init__(self, cfg):
        super(WHAMSAHNet, self).__init__()
        
    def forward(self, x):
        return x