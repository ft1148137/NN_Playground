import torch.nn as nn
import torch
import math

class PositoinalEncoding(nn.Module):
    def __init__(self,max_len,d_model,device):
        super(PositoinalEncoding,self).__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device)
        pos = torch.arange(0,max_len,device=device)
        i = torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2] = torch.sin(pos/(10000**(i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(i/d_model)))
        pass
    
    def forward(self,x):
        return self.encoding[:len(x),:]