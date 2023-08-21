import torch.nn as nn
import torch
import numpy as np
import math

class SingleAttention(nn.Module):
    def __init__(self,D_k,D_model):
        super(SingleAttention,self).__init__()
        self.d_k = D_k
        self.w_q = nn.Linear(D_model,D_k)
        self.w_k = nn.Linear(D_model,D_k)
        self.w_v = nn.Linear(D_model,D_k)
        self.softmax = nn.Softmax(dim = -1)
        pass
    
    def forward(self,q,k,v,mask = False):
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        l1 = q.mm(k.transpose(0,1))/math.sqrt(self.d_k)
        if not mask:
            l1 = l1.masked_fill(torch.tril(torch.ones_like(l1),diagonal=0) == 0,-1e10)
        l2 = self.softmax(l1)
        out = l2.mm(v)
        return out
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,D_k,D_model):
        super(MultiHeadAttention,self).__init__()
        self.single_attention = SingleAttention(D_k,D_model)
        self.no_single_attention = int(D_model/D_k)
        pass
    
    def forward(self,q,k,v,mask = False):
        return torch.cat([self.single_attention(q,k,v,mask) for i in range(self.no_single_attention)],1)
    
    
    
x = torch.randn(5,512)
model = MultiHeadAttention(64,512)
y = model(x,x,x)      
print(y.shape)  