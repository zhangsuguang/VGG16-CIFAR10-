import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import VGG16实现cifar10分类
from PIL import Image
cifar10_classes = ('plane','car','bird','cat','deer',
                   'dog','frog','horse','ship','truck')
transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),#以输入图像img的中心作为中心点进行指定size的crop操作
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])  # 概率性修改原图
def predict(img_path):
    net = VGG16实现cifar10分类.VGG('VGG16')

    modelpath = 'F:\PythonProject\checkpoint\ew_model\cifar10_epoch_5.ckpt'
    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda()
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.cuda()
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    print('this picture maybe :', cifar10_classes[predicted[0]])
if __name__ =='__main__':
    predict('D:\火狐\火狐下载\cat1.jpg')



'''
RuntimeError: size mismatch, m1: [1 x 25088], m2: [512 x 512] 
卷积池化结束后图片的size为[1 x 25088]  与fc层的size[512 x 512] 不匹配
将原来前向传播的这行代码改为 x = x.view(-1,512)
'''
