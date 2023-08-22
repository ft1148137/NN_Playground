import torch.nn as nn
import torch
import math
from mulit_head_attention import MultiHeadAttention
from feed_forward import FeedForward
class Decoder(nn.Module):
    def __init__(self,D_k,D_model,D_ff):
        super(Decoder,self).__init__()
        self.masked_mulit_head_attention = MultiHeadAttention(D_k,D_model)
        self.mulit_head_attention = MultiHeadAttention(D_k,D_model)
        self.feed_forward = FeedForward(D_ff,D_model)
        self.layer_norm_1 = nn.LayerNorm(D_model)
        self.layer_norm_2 = nn.LayerNorm(D_model)
        self.layer_norm_3 = nn.LayerNorm(D_model)

    def forward(self,x,q,k):
        l1 = self.layer_norm_1(self.masked_mulit_head_attention(x,x,x,mask = True) + x)
        l2 = self.layer_norm_2(self.mulit_head_attention(q,k,x)+l1)
        l3 = self.layer_norm_3(self.feed_forward(l2) + l2)
        return l3
    
x = torch.randn(5,512)
model = Decoder(64,512,2048)
model.eval()
print(sum(p.numel() for p in model.parameters()))  
y = model(x,x,x) 
print(model)  
print(y.shape)   
torch.save(model,'save.pt')
