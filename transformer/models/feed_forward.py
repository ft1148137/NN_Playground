import torch.nn as nn
import torch
import math

class FeedForward(nn.Module):
    def __init__(self,D_ff,D_model,drop_out = 0.1):
        super(FeedForward,self).__init__()
        self.out = nn.Sequential(
        nn.Linear(D_model,D_ff),
        nn.ReLU(),
        nn.Dropout(p=drop_out),
        nn.Linear(D_ff,D_model)
        )
        pass
    
    def forward(self,x):
        return self.out(x)
##test net           
x = torch.randn(5,512)
model = FeedForward(2048,512)
model.eval()
print(sum(p.numel() for p in model.parameters()))  
y = model(x) 
print(model)  
print(y.shape)   
