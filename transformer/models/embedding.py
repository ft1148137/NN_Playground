import torch.nn as nn
import torch
from positional_encoding import PositoinalEncoding
class Embedding(nn.Module):
    def __init__(self,txt_len,d_model,max_len,drop_prob = 0.1):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(txt_len,d_model,padding_idx=1)
        self.positoinal_encoding = PositoinalEncoding(max_len,d_model,"cpu")
        self.drop_out = nn.Dropout(p = drop_prob)
        pass
    
    def forward(self,x):
        return self.drop_out(self.embedding(x) + self.positoinal_encoding(x))