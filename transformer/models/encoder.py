import torch.nn as nn
import torch
from mulit_head_attention import MultiHeadAttention
from feed_forward import FeedForward
class Encoder(nn.Module):
    def __init__(self,D_k,D_model,D_ff):
        super(Encoder,self).__init__()
        self.multi_head_attention = MultiHeadAttention(D_k,D_model)
        self.layer_norm_1 = nn.LayerNorm(D_model)
        self.feed_forward = FeedForward(D_ff,D_model)
        self.layer_norm_2 = nn.LayerNorm(D_model)
        
    def forward(self,x):
        L1 = self.layer_norm_1((self.multi_head_attention(x,x,x) + x))
        L2 = self.layer_norm_2((self.feed_forward(L1) + L1))
        return L2
    
    
x = torch.randn(5,512)
model = Encoder(64,512,2048)
model.eval()
# print(sum(p.numel() for p in model.parameters()))  
y = model(x) 
print(model)  
# print(y.shape)   
torch.save(model,'save.pt')
        
        
