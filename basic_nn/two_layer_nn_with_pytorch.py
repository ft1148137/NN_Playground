
    
import numpy as np
import torch
N, D_in, H, D_out = 64, 1000,100, 10

x= torch.randn(N,D_in)
y= torch.randn(N,D_out)
w1 = torch.randn(D_in, H,requires_grad=True)
w2 = torch.randn(H, D_out,requires_grad=True)

learning_reate = 1e-6

for iter in range(200):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    loss = (y_pred - y).pow(2).sum()
    print("iter:",iter,"loss:",loss.item())
    
    loss.backward()
    with torch.no_grad():
        w1 -= learning_reate * w1.grad
        w2 -= learning_reate * w2.grad 
        w1.grad.zero_()
        w2.grad.zero_()
    