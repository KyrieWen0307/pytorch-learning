import torch
import torchvision
from model_save import *
# 加载方式1：加载完整模型
# PyTorch 2.6+ 默认 weights_only=True，加载完整类对象必须显式传入 weights_only=False
# model = torch.load('models/vgg16_method1.pth', weights_only=False)
# print('方式1加载成功（完整模型）：')
# print(model)

#方式2
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load('models/vgg16_method2.pth'))
# print(vgg16)

#陷阱1
model = torch.load('models/kyrie_method1.pth',weights_only=False)
print(model)