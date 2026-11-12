from torch import nn
import torch


class Yuhao(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input+1
        return output

Yuhao = Yuhao()
x = torch.tensor(1.0)
print(Yuhao(x))