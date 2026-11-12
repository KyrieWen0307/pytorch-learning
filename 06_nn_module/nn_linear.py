import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Kyrie_linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output
kyrie = Kyrie_linear()

for data in dataloader:
    imgs,targets = data
    output = torch.flatten(imgs)
    output = kyrie(output)
    print(output.shape)