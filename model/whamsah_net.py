"""
7.0: like 6.0 but without context to reduce inference time
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
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=8, stride=2, dilation=1, padding=3)#, padding='same')
        self.pool = nn.AvgPool1d(kernel_size=8, stride=2, ceil_mode=True)
        #self.BN = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
    def forward(self,x, skip=True):
        c = self.conv(x) #out is N - 8  + 1
        #x = self.BN(c)
        x = self.act(c)
        if skip:
            return x, c
        return x

class WHDecLayer(nn.Module):
    def __init__(self, in_c, out_c, hasAttention=False, last=False):
        super(WHDecLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_c, out_c, kernel_size=8, stride=2, dilation=1, padding=3)
        #self.BN = nn.BatchNorm1d(out_c)
        self.last=last
        self.act = nn.GELU()
        self.hasAttention = hasAttention
        if hasAttention:
            self.innerAtt = nn.MultiheadAttention(in_c, 4, batch_first=True)
        self.innerConv = nn.Conv1d(in_c, 2*in_c, kernel_size=5, stride=1, dilation=1, padding=2)
        self.innerGLU = nn.GLU(dim=1)

    def forward(self,x, skip=None):
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
        x = self.conv(x)
        #x = self.BN(x)
        if self.last:
            return x
        x = self.act(x)
        return x


class ConvMidLayer(nn.Module):
    def __init__(self, in_c,segment_l, depth, contextSize=5,):
        super(ConvMidLayer, self).__init__()
        assert depth is not None, "Depth must be specified in ConvMidLayer"
        assert segment_l is not None, "Segment length must be specified in ConvMidLayer"
        self.in_c = in_c
        self.in_len = segment_l / (2**depth) #length of the input segment in samples
        self.GN = nn.GroupNorm(in_c, in_c)
        self.tRNN = nn.RNN(in_c, in_c*2, batch_first=True, dropout=0.1) #TIME RNN - features are in the time dimension
        #NOT bidirectional
        self.ff1 = nn.Linear(in_c*2, in_c)
        
        self.GN2 = nn.GroupNorm(in_c, in_c)
        self.cRNN = nn.RNN(in_c, in_c*2, batch_first=True, dropout=0.1) #CHANNEL RNN - features are in the channel dimension
        #NOT bidirectional
        self.ff2 = nn.Linear(in_c*2, in_c)

        #self.att = nn.MultiheadAttention(in_c, 1, dropout=0.1, batch_first=True,
        #                                 kdim=in_c, vdim=in_c)
        self.contextSize = contextSize
        
        #self.previousInput = None
        self.context = []
    def forward(self, input,debug=False):
        """
        #self.updateContext(input.clone().detach(), updateComputed=False)
        #q = self.innerFQ.getQueue()
        if len(self.context) == 0:
            print(f"Initializing context with shape {input.shape}") if debug else None
            for i in range(self.contextSize-1):
                self.context.append(torch.zeros_like(input, requires_grad=False))
        self.context.append(input.detach().clone())
        if len(self.context) > self.contextSize:
            self.context.pop(0)
        #move all elements in q to the device of input
        #now the array is always full
        h = torch.cat(self.context, dim=2)
        if h.device != input.device:
            h = h.to(input.device)
        """
        x = self.GN(input)
        x = rearrange(x, 'b c (d n) -> b (c n) d', d=self.in_c, n=int(self.in_len/self.in_c))
        x = self.tRNN(x)[0]
        x = self.ff1(x)
        x = rearrange(x, 'b (c n) d -> b c (d n)', c=self.in_c, d=self.in_c, n=int(self.in_len/self.in_c))
        input = x+input #skip
        #print(h.shape) #4, 40960, 16
        #x = rearrange(input, 'b (c n) d -> b c (d n)', c=self.in_c, d=self.in_c, n=int(self.in_len*self.contextSize/self.in_c))
        #g = rearrange(g, 'b c t -> b t c')
        x = self.GN2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.cRNN(x)[0]
        x = self.ff2(x)
        x = rearrange(x, 'b t c -> b c t')
        #h = rearrange(h, 'b c t -> b t c') #unneeded, h is already [b, t, c]
        input = x+input
        return input
    def updateContext(self, input,updateComputed=False):
        self.innerFQ.push(input,update=updateComputed) #the first execution will be a bit pricier 
        #as it needs to create some arrays
    def clearQueue(self):
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
                self.DECLAYERS.append(WHDecLayer(self.in_c, self.in_c//self.growth, last=True))
            else:
                self.DECLAYERS.append(WHDecLayer(self.in_c, self.in_c//self.growth, hasAttention=False))
            self.in_c = self.in_c//self.growth
        
    def clearQueue(self):
        try:
            self.midConvLayer.clearQueue()
            #self.MML.clearQueue()
        except:
            pass
    def forward(self, x):
        #normalization should be done OUTSIDE the net
        #print(x.shape) #4, 2, 65536 -- B, C, T 
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x-mean)/(std+EPSILON)
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
        x = (x*(std+EPSILON)) + mean
        return x
    


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