import os
import torchvision
import torch
from torch import nn
from torch.nn import Conv2d

os.makedirs('models', exist_ok=True)

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1：保存完整模型（结构 + 参数）
torch.save(vgg16, 'models/vgg16_method1.pth')
print('方式1保存成功 -> models/vgg16_method1.pth')

# 保存方式2（官方推荐）：只保存参数字典 state_dict
torch.save(vgg16.state_dict(), 'models/vgg16_method2.pth')
print('方式2保存成功 -> models/vgg16_method2.pth')

#陷阱1
class Kyrie(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(3,64,kernel_size=3)

    def forward(self,input):
        x = self.conv1(input)
        return x

kyrie = Kyrie()
torch.save(kyrie,'models/kyrie_method1.pth')