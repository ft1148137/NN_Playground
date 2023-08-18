
    
import numpy as np
import torch
import torch.nn as nn
N, D_in, H, D_out = 64, 1000,100, 10

x= torch.randn(N,D_in)
y= torch.randn(N,D_out)

learning_reate = 1e-4
model = nn.Sequential(
    nn.Linear(D_in,H),
    torch.nn.ReLU(),
    nn.Linear(H,D_out)
)
loss_fn = nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(),lr = learning_reate)
for iter in range(200):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(iter, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()