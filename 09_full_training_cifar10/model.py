import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch




class Kyrie_CIFAR10(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,input):
        output = self.model(input)
        return output

if __name__ == '__main__':
    kyrie = Kyrie_CIFAR10()
    input = torch.ones((64,3,32,32))
    output = kyrie(input)
    print(output.shape)