import torch
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
# input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Kyrie(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu1 = ReLU(inplace=False) 
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

writer = SummaryWriter('logs/sigmoid')
kyrie = Kyrie()
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('sigmoid_input',imgs,step)
    output = kyrie(imgs)
    writer.add_images('sigmoid_output',output,step)
    step+=1

writer.close()