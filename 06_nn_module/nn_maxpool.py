import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)


class Kyrie(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool(input)
        return output

kyrie = Kyrie()

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('maxpool_input',imgs,step)
    output = kyrie(imgs)
    writer.add_images('maxpool_output',output,step)
    step+=1
writer.close()