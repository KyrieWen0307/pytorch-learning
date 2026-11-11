from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open(r'dataset\data\train\ants_image\0013035.jpg')
print(img)

#totensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image('totensor',img_tensor)

#normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize',img_norm,2)

#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(type(img_resize))
writer.add_image('resize',img_resize,0)

#resize2
trans_resize2=transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize2,trans_totensor])
img_resize2=trans_compose(img)
writer.add_image('resize2',img_resize2,0)

#randomcrop
trans_random=transforms.RandomCrop(512)
trans_compose2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_random=trans_compose2(img)
    writer.add_image('random',img_random,i)
writer.close()