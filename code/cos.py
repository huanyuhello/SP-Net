
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.ones(5,20)
b = torch.ones(5,20)*4
target = torch.tensor([1])
cos1 = F.cosine_embedding_loss(a,b, torch.tensor([1]),  margin=0)
cos2 = F.cosine_embedding_loss(a,b, torch.tensor([-1]),  margin=0)

print(cos1, cos2)

print(torch.cosine_similarity(a,b,dim=-1), torch.cosine_similarity(a,-b,dim=-1))


up = torch.nn.Upsample(size=400,  mode='nearest')
down = torch.nn.Upsample(size=2,  mode='nearest')

pool = torch.nn.AdaptiveAvgPool1d(10)

a = a.unsqueeze(1)
print(up(a).shape, down(a).shape, pool(a).shape)