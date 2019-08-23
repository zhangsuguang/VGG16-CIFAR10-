#####################################
# 第一步 载入数据
#####################################
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
#使用torchvision可以方便的下载Cifar10数据集 而torchvision下载的数据集为[0,1]的
#PILImage格式  我们需要将张量Tensor归一化到[-1,1]

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
                                transforms.ToTensor(), #将PILImage转换为张量
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])
train_dataset = dsets.CIFAR10(root='/ml/pycifar',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=transform,
                            download=True)
test_dataset = dsets.CIFAR10(root='/ml/pycifar',
                           train=False,# 选择测试集
                           transform=transform,
                           download=True)
trainloader = DataLoader(train_dataset,
                         batch_size=4,  #  每个batch载入的图片数量
                         shuffle=True,
                         num_workers=2)  #载入训练数据所需的子任务数
testloader = DataLoader(test_dataset,
                         batch_size=4,  #  每个batch载入的图片数量
                         shuffle=False,
                         num_workers=2)  #载入训练数据所需的子任务数
cifar10_classes = ('plane','car','bird','cat','deer',
                   'dog','frog','horse','ship','truck')
# #####################################
# # 查看训练数据
# # 备注：该部分代码可以不放入主函数
# #####################################
# import numpy as np
# if __name__ == '__main__':
#     dataiter = iter(trainloader) #从训练数据中随机取一些数据
#     images,labels = dataiter.next()
#     print(images.shape)
#     torchvision.utils.save_image(images[1],'test.jpg') #随机保存一张images里面的一张图片看看
#     for j in cifar10_classes:
#         print(labels[0])

#####################################
# 第二步 构建卷积神经网络
#####################################

cfg = {'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']}

class VGG(nn.Module):
    def __init__(self,net_name):
        super(VGG,self).__init__()
        #构建卷积网络的卷积层和池化层 最终输出命名features，原因是通常认为经过这些惭怍的输出
        #为包含图像空间信息的特征层
        self.features = self._make_layers(cfg[net_name])

        # 构建卷积层之后的全连接层及分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),#fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),#fc2
            nn.ReLU(True),
            nn.Linear(512, 10),  # fc3 最终cifar10 输出的是10类
        )
        #初始化权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d): #判断某个模块是否是某种类型
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)  # 前向传播时先经过卷积层和池化层
        x = x.view(-1,512)
        x = self.classifier(x)
        return x

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
net = VGG('VGG16')
net = net.cuda() #使用gpu
#####################################
# 第三步 定义损失函数和优化方法
#####################################
criterion = nn.CrossEntropyLoss()   #定义损失函数 交叉熵
optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9) # 定义优化方法 随机梯度下降

#####################################
# 第四步 卷积神经网络的训练
#####################################

def train():
    for epoch in range(5):
        train_loss = 0.0
        for batch_idx,data in enumerate(trainloader,0):
            #初始化
            inputs,labels = data  #获取数据
            optimizer.zero_grad()  # 先将梯度设为0

            #优化过程
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)  #将数据输入到网络 得到第一轮网络前向传播的预测结果outputs
            loss = criterion(outputs,labels) #预测结果outputs和labels通过之前定义的交叉熵计算损失
            loss.backward()   #误差反向传播
            optimizer.step()    #随机梯度下降方法（之前定义的）优化权重

            #查看网络训练状态
            train_loss += loss.item()
            if batch_idx % 2000 ==1999:  #每迭代2000个batch打印一次以查看当前网络的收敛情况
                print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,train_loss/2000))
                train_loss = 0.0
        print('Saving epoch %d model'%(epoch+1))
        state = {
            'net':net.state_dict(),  # 模型的网络参数
            'epoch':epoch + 1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/cifar10_epoch_%d.ckpt' %(epoch+1))
    print('Finishef Training')

def test():
    net = VGG('VGG16')
    modelpath = 'F:\PythonProject\checkpoint\ew_model\cifar10_epoch_5.ckpt'
    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()
    print('Accuracy of the network on the 10000 test images: %d %% '%((100 * correct)/total))
    ######################################
    #分别查看每个类的预测结果
    ######################################
    class_correct = list(0. for i in range(10))# 定义一个存储每类中测试正确的个数的 列表，初始化为0
    class_total = list(0. for i in range(10))  # 定义一个存储每类中测试总数的个数的 列表，初始化为0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):  # 因为每个batch都有4张图片，所以还需要一个4的小循环
                label = labels[i]   # 对各个类的进行各自累加
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            cifar10_classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    test()


'''
  在加载模型测试的时候 出现并解决的问题：
  1：net.load_state_dict('net_params.pkl')  #你训练好的模型文件好像字典键值有很多个，包括epoch等，但我们只需要模型参数文件。
                                            报错的原因是因为载入模型文件的键值太多了。pytorch识别不了。
  2. modelpath = 'F:\PythonProject\checkpoint\ew_model # 这样会有权限错误 因为打开的是文件夹 并非文件，
                                                            所以要在后面加上要打开的文件名

'''