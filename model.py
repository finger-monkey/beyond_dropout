import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x
import math
def CREP(H,W,sl=0.02, sh=0.4, r1=0.3):#该函数仅仅用来随机生成一个合理的擦除块的位置，不进行随机擦除操作

    for attempt in range(100):
        area = H * W
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < W and h < H:
            x1 = random.randint(0, H - h)
            y1 = random.randint(0, W - w)
            return (h,w,x1,y1)

# import random
# randNum=512 #随机对特征图的8个通道做失活处理
#
# def RandChannelDropout(x):
#     #假设当前层的特征图的通道数为64，则要从 0-63中随机选出8个通道编号
#     a = []
#     for i in range(0,randNum):
#         a.append(random.randint(0,2047))
#     # print("a=",a)
#     # print("a[2]=", a[2])
#     # a[0]=0 #控制变量以进行可控分析
#     #-------------为了进行可控分析实验，对数据集的加载暂时改为非乱序---------------
#
#     #对随机挑选出来的通道进行“失活”  >>torch.Size([128, 64, 128, 64]) 64个通道
#     #以一定的概率来对整个批组来进行 随机通道进行“失活” 处理
#     p = random.random()
#     if p < 0.5:
#         for i in range(0,32): #对整个批组的n个样本逐一操作
#             for j in range(0, randNum): #对该样本的randNum个通通做处理
#                 x[i][j] = 0  #对通道进行“失活”
#     return x

import random
batchsize = 8
randNum=32 #随机对特征图的randNum个通道做失活处理                torch.Size([32, 64, 128, 64]) 64个通道
def RandChannelDropout(x):
    #假设当前层的特征图的通道数为64，则要从 0-63中随机选出8个通道编号
    a = []
    for i in range(0,randNum):
        a.append(random.randint(0,63))
    # print("a=",a)
    # print("a[2]=", a[2])
    # a[0]=0 #控制变量以进行可控分析
    #-------------为了进行可控分析实验，对数据集的加载暂时改为非乱序---------------

    #对随机挑选出来的通道进行“失活”  >>torch.Size([128, 64, 128, 64]) 64个通道
    #以一定的概率来对整个批组来进行 随机通道进行“失活” 处理
    p = random.random()
    if p < 0.15:
        for i in range(0,batchsize): #对整个批组的n个样本逐一操作
            for j in range(0, randNum): #对该样本的randNum个通通做处理
                # x[i][j] = 0  #对通道进行“失活”
                h,w,x1,y1 = CREP(H=128,W=64)
                # print('h=',h,"w=",w,"x1=",x1,'y1=',y1)
                x[i][a[j]][y1:y1 + h][x1:x1 + w] = random.random()
    return x

# randNum2=32 #随机对特征图的randNum个通道做失活处理                self.model.layer1(x)#torch.Size([32, 256, 64, 32])
# def RandChannelDropout2(x):
#     #假设当前层的特征图的通道数为64，则要从 0-63中随机选出8个通道编号
#     a = []
#     # randNum = (x[0].size()[0])/2
#     for i in range(0,randNum2):
#         # print("x[0].size()[0]=",x[0].size()[0])
#         a.append(random.randint(0, x[0].size()[0]-1))
#     p = random.random()
#     if p < 0.15:
#         for i in range(0,32): #对整个批组的n个样本逐一操作
#             for j in range(0, randNum): #对该样本的randNum个通通做处理
#                 # x[i][j] = 0  #对通道进行“失活”
#                 h,w,x1,y1 = CREP(H=64,W=32)
#                 # print('h=',h,"w=",w,"x1=",x1,'y1=',y1)
#                 x[i][a[j]][y1:y1 + h][x1:x1 + w] = random.random()
#     return x

# import random
# randNum=64 #随机对特征图的randNum个通道做失活处理
# def RandChannelDropout(x):
#
#     p = random.random()
#     if p < 0.05:
#         for i in range(0,32): #对整个批组的n个样本逐一操作
#             for j in range(0, randNum): #对该样本的randNum个通通做处理
#                 # x[i][j] = 0  #对通道进行“失活”
#                 h,w,x1,y1 = CREP(H=128,W=64)
#                 # h = int(h/2)
#                 # w = int(w/2)
#                 # print('h=',h,"w=",w,"x1=",x1,'y1=',y1)
#                 # x[i][j][y1:y1 + h][x1:x1 + w] = 0.4914
#                 x[i][j][y1:y1 + h][x1:x1 + w] = random.random()
#     return x

print('---------------market-CRE_P015_RC-32（random value,all Channel=64,Front ）------------------------')
# print('---------------market-all------------------------')
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        # torch.Size([32, 3, 256, 128])
        x = self.model.conv1(x)#torch.Size([32, 64, 128, 64]) 32为batchsize,64为通道数
        # print("x=\n",x[0][0])
        # x = RandChannelDropout(x)
        # print("RandChannelDropout->x=\n", x[0][0])
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)#torch.Size([32, 64, 64, 32])
        # x = RandChannelDropout2(x)
        x = self.model.layer1(x)#torch.Size([32, 256, 64, 32])
        x = self.model.layer2(x)# torch.Size([32, 512, 32, 16])
        x = self.model.layer3(x)#torch.Size([32, 1024, 16, 8])
        x = self.model.layer4(x)#torch.Size([32, 2048, 8, 4])
        x = self.model.avgpool(x)   #torch.Size([32, 2048, 1, 1])
        # print("x.size:", x.size())
        # x = RandChannelDropout(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:,:,i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net(751, stride=1, ibn=True)
    #net = ft_net_swin(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)
