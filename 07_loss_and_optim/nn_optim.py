import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Kyrie_CIFAR10(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss=CrossEntropyLoss()
kyrie = Kyrie_CIFAR10()
optim = torch.optim.SGD(kyrie.parameters(),lr=0.05)
for epoch in range(20):
    running_loss =0.0
    for data in dataloader:
        imgs,targets = data
        output = kyrie(imgs)
        result_loss = loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss.item()
    print(running_loss )