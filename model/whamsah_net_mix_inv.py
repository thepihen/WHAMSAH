"""
9.0: 7 + applenet-like sconv layers. FIXED
"""

import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

#from 
#model augmentations:
#window type
#stereo
#mixup
EPSILON = 1e-8

class WHEncLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(WHEncLayer, self).__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, dilation=1, padding=1)#, padding='same')
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        #self.BN = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
    def forward(self,x, skip=True):
        c = self.conv(x) #out is N - 8  + 1
        #x = self.BN(c)
        x = self.act(c)
        x = self.pool(x)
        if skip:
            return x, c
        return x

class WHDecLayer(nn.Module):
    def __init__(self, in_c, out_c, hasAttention=False, last=False):
        super(WHDecLayer, self).__init__()
        self.convt = nn.ConvTranspose1d(in_c, in_c, kernel_size=4, stride=2, dilation=1, padding=1)
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, dilation=1, padding=1)
        #self.BN = nn.BatchNorm1d(out_c)
        self.last=last
        self.act = nn.GELU()
        self.hasAttention = hasAttention
        if hasAttention:
            self.innerAtt = nn.MultiheadAttention(in_c, 4, batch_first=True)
        self.innerConv = nn.Conv1d(in_c, 2*in_c, kernel_size=1, stride=1, dilation=1)
        self.innerGLU = nn.GLU(dim=1)

    def forward(self,x, skip=None):
        x = self.convt(x)
        if skip is not None:
            if self.hasAttention:
                skip = rearrange(skip, 'b c t -> b t c')
                y,_ = self.innerAtt(skip,skip,skip)
                skip = skip+y
                skip = rearrange(skip, 'b t c -> b c t')
            skip = self.innerConv(skip)
            skip = self.innerGLU(skip)
            skip = self.act(skip)
            x += skip
        #x = self.BN(x)
        x = self.conv(x)
        if self.last:
            return x
        x = self.act(x)
        return x

class SConv(nn.Module):
    def __init__(self,in_c, D):
        super(SConv, self).__init__()
        self.c1 = nn.Conv1d(in_c, in_c, kernel_size=1, stride=1, dilation=1)
        self.GN1 = nn.GroupNorm(in_c, in_c)
        self.GN2 = nn.GroupNorm(in_c, in_c)
        
        self.pre1 = nn.PReLU()
        self.c2 = nn.Conv1d(in_c, in_c, kernel_size=3, stride=1, dilation=D, padding=D)
        self.pre2 = nn.PReLU()
    def forward(self,input):
        x = self.c1(input)
        x = self.pre1(x)
        x = self.GN1(x)
        x = self.c2(x)
        x = self.pre2(x)
        x = self.GN2(x)
        x = x + input
        return x 

class ConvMidLayer(nn.Module):
    def __init__(self, in_c,segment_l, depth, R=6, D=9, contextSize=5,):
        super(ConvMidLayer, self).__init__()
        assert depth is not None, "Depth must be specified in ConvMidLayer"
        assert segment_l is not None, "Segment length must be specified in ConvMidLayer"
        self.in_c = in_c
        self.in_len = segment_l / (2**depth) #length of the input segment in samples
        self.parts = nn.ModuleList()
        for i in range(R):
            innerParts = nn.ModuleList()
            for j in range(D):
                innerParts.append(SConv(in_c, 2**j))
            self.parts.append(innerParts)
        self.context = []
    def forward(self, x,debug=False):
        for i in range(len(self.parts)):
            for j in range(len(self.parts[i])):
                x = self.parts[i][j](x)
        return x
    def updateContext(self, input,updateComputed=False):
        self.innerFQ.push(input,update=updateComputed) #the first execution will be a bit pricier 
        #as it needs to create some arrays
    def clearQueue(self):
        return
        #self.RNN_hidden = None
        self.context = []
        #self.innerFQ.clearQueue()

class WHAMSAHNet(nn.Module):
    def __init__(self, cfg):
        super(WHAMSAHNet, self).__init__()
        self.depth = cfg['model']['depth']
        self.growth = cfg['model']['growth']
        self.in_c = 2
        self.out_c = 2 # 2 speakers * 1 part (vocals only)
        self.ENCLAYERS = nn.ModuleList()
        self.DECLAYERS = nn.ModuleList()
        for i in range(self.depth):
            self.ENCLAYERS.append(WHEncLayer(self.in_c, self.in_c*self.growth))
            self.in_c = self.in_c*self.growth
        self.midConvLayer = ConvMidLayer(self.in_c, segment_l=cfg['separator']['chunk'], depth=self.depth)
        #self.MML = MagicalMidLayer(self.in_c, self.in_c)
        for i in range(self.depth):
            if i==self.depth-1:
                self.DECLAYERS.append(WHDecLayer(self.in_c, self.in_c//self.growth, hasAttention=False, last=True))
            else:
                self.DECLAYERS.append(WHDecLayer(self.in_c, self.in_c//self.growth, hasAttention=False))
            self.in_c = self.in_c//self.growth
        #self.sigma = nn.Sigmoid()
    def clearQueue(self):
        return
        try:
            self.midConvLayer.clearQueue()
            #self.MML.clearQueue()
        except:
            pass
    def forward(self, inp):
        #normalization should be done OUTSIDE the net
        #print(x.shape) #4, 2, 65536 -- B, C, T 
        mean = inp.mean(dim=2, keepdim=True)
        std = inp.std(dim=2, keepdim=True)
        x = (inp-mean)/(std+EPSILON)
        #inp = (inp-mean)/(std+EPSILON)
        skips = []
        for i in range(self.depth):
            if i != self.depth-1:
                x, skip = self.ENCLAYERS[i](x)
                skips.append(skip)
            else:
                x = self.ENCLAYERS[i](x, False)
        #print(f"1: {x.shape}")
        #x = self.MML(x)
        x = self.midConvLayer(x)
        for i in range(self.depth):
            if i==0:
                x = self.DECLAYERS[i](x)
            else:
                x = self.DECLAYERS[i](x, skips[-i])
        #print(f"2: {x.shape}")
        #x = self.sigma(x)
        #x = x * inp
        x = (x*(std+EPSILON)) + mean
        return inp - x
    


class FrameQueue:
    def __init__(self, frameSize, hopSize=None):
        self.queue = []
        self.maxSize = 5
        self.hopsize=hopSize
        
    def push(self, frame, update=True):
        if len(self.queue) == 0:
            self.setup(frame.shape)
            for i in range(self.maxSize-1):
                self.queue.append(torch.zeros(frame.shape)) #needed for getQueue
        if len(self.queue) == self.maxSize:
            self.queue.pop(0) #pos 0 always has the oldest element
        #self.queue.append(frame.detach())
        self.queue.append(frame)
        if update:
            self.updateComputed()
    def updateComputed(self):
        for i, frame in enumerate(self.queue):
            try:
                self.computedQ[..., i*self.hop:i*self.hop+self.tsize] += frame.to(self.computedQ.device)
            except:
                print(f"Index error while updating computedQ for frame index {i}")
                continue
    def setup(self, framesize):
        B, C, T = framesize
        if self.hopsize is not None:
            self.hop = self.hopsize
        else:
            self.hop = T
        self.computedQ = torch.zeros((B, C, int(T*(self.hop/T)*self.maxSize)))
        self.size = framesize
        self.tsize = T
        self.ROPE = RotaryPositionalEmbeddings(dim=256,max_seq_len=T*self.maxSize)
            
    def getComputed(self):
        return self.computedQ
    def getLastFrame(self):
        return self.queue[-1]
    def getQueue(self):
        return self.queue
    def applyROPE(self):
        #apply rotary positional embeddings to the computedQ
        pass
    def getROPEComputed(self):
        return self.applyROPE()
    
    def clearQueue(self):   
        self.queue = []
        self.computedQ = None