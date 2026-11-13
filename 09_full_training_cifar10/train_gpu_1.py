import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch
#from model import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10('./dataset',train=True,download=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

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
    
kyrie = Kyrie_CIFAR10()
if torch.cuda.is_available():   
    kyrie = kyrie.cuda()
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():   
    loss_fn = loss_fn.cuda()
learning_rate = 0.02
optimizer = torch.optim.SGD(kyrie.parameters(),lr=learning_rate)


total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter('./logs/full_train_test')

for i in range (epoch):
    print(f'——————第{i+1}轮训练开始——————')
    kyrie.train()
    for data in train_dataloader:
        imgs,targets = data
        if torch.cuda.is_available():   
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = kyrie(imgs)
        loss = loss_fn(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step +=1
        if total_train_step%100 == 0:
            print(f'训练次数为{total_train_step},此时loss={loss.item()}')
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    kyrie.eval()
    total_loss = 0
    total_acc =  0
    with torch.no_grad():
        for data in test_dataloader:
                imgs,targets = data
                if torch.cuda.is_available():   
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = kyrie(imgs)
                loss = loss_fn(outputs,targets)
                total_loss += loss.item()
                acc = (outputs.argmax(1) == targets).sum()
                total_acc += acc
    print(f'total_loss={total_loss}')
    print(f'total_acc={total_acc/len(test_data)}')
    writer.add_scalar('test_loss',total_loss,total_test_step)
    writer.add_scalar('test_acc',total_acc/len(test_data),total_test_step)
    total_test_step+=1

writer.close()
torch.save(kyrie,'models/kyrie_cifar10.pth')










