import torchvision
from PIL import Image
from model import Kyrie_CIFAR10
import torch


image_path= r'dataset\dog.png'
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
model = torch.load('models/kyrie_cifar10.pth', weights_only=False)

image = transform(image)
print(image.shape)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))

# 1. 把原始打分 (Logits) 用 Softmax 转成加起来等于 100% 的概率分布
probs = torch.softmax(output, dim=1)
print(f"训练后模型认为它是狗(5)的概率: {probs[0][5].item() * 100:.2f}%")

# 2. 造一个完全没训练过的“野生模型”同台PK
untrained_model = Kyrie_CIFAR10()
untrained_model.eval()
with torch.no_grad():
    untrained_out = untrained_model(image)
    untrained_probs = torch.softmax(untrained_out, dim=1)
print(f"未训练野生模型的原始输出: {untrained_out}")
print(f"未训练模型认为它是狗(5)的概率: {untrained_probs[0][5].item() * 100:.2f}% (纯靠蒙，约10%)")