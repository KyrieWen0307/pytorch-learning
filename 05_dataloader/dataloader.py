import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#测试数据集
test_data=torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)
writer = SummaryWriter('logs/dataloader')
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,target = data
        # print(imgw.shape)
        # print(target)
        writer.add_images(f'{epoch}',imgs,step)
        step+=1

writer.close()