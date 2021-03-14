import torch

a=torch.randn(32,2,16)
print(a)
b=torch.norm(a,dim=-1)
print(b)