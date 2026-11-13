import torchvision
from torch.utils.data import DataLoader
from torch import nn



train_data = torchvision.datasets.CIFAR10('./dataset',train=True,download=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')
vgg16_true.add_module('add_linear',nn.Linear(1000,10))
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_true)
print (vgg16_false)