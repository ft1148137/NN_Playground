import torch.nn as nn
import torch
import math

class FeedForward(nn.Module):
    def __init__(self,D_ff,D_model):
        super(FeedForward,self).__init__()
        self.w1 = nn.Linear(D_model,D_ff)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(D_ff,D_model)
        pass
    
    def forward(self,x):
        return self.w2(self.relu(self.w1(x)))
        
x = torch.randn(5,512)
model = FeedForward(2048,512)
model.eval()
print(sum(p.numel() for p in model.parameters()))  
y = model(x) 
print(model)  
print(y.shape)   
