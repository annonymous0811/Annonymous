import torch
import torch.nn as nn
import numpy as np 
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

from config import dataset_path
import imageio

# cc_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_15c8e59fdfafb3deefc80c5a9e8a42d0.png')
# mlo_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_83be060130997ca7b67b3979978a5d29.png')

# input_tensor = torch.Tensor(mlo_img)
# input_tensor = torch.movedim(input_tensor, 2, 0).unsqueeze(0)

class Fusion_ResNet(nn.Module):
    def __init__(self, backbone, model_type, aggregation, num_classes=2, dropout = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.backbone = backbone
        self.model_type = model_type
        self.fc_expand = 1

        early_layers_cc = []
        early_layers_mlo = []
        last_layers = []
        
        child_counter = 0
        if model_type == "PreF":
            for child in self.backbone.children():
                if child_counter < 0:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            
            out_n = 3
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n

        if model_type == "EF":
            for child in self.backbone.children():
                if child_counter <= 3:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            
            out_n = early_layers_cc[0].out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n

        if model_type == "MF":
            for child in self.backbone.children():
                if child_counter <= 5:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1

            out_n =  early_layers_cc[-1][0].conv1.out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n * self.backbone.block.expansion

        if model_type == "LF":
            for child in self.backbone.children():
                if child_counter <= 7:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            out_n =  early_layers_cc[-1][0].conv1.out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n * self.backbone.block.expansion

        if model_type == "PostF":
            for child in self.backbone.children():
                if child_counter <= 11:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            out_n =  3
            out_fc = early_layers_cc[-1].out_features
            if self.aggregation == 'cat':
                self.fc_expand = 2
            self.out_dim =  out_n * self.backbone.block.expansion
            
        self.conv_avg = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False)
        self.conv_ccat = nn.Conv2d(self.out_dim*2 , self.out_dim , kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False) 
        self.bn = nn.BatchNorm2d(self.out_dim)

        self.early_layers_cc = nn.Sequential(*early_layers_cc)
        self.early_layers_mlo = nn.Sequential(*early_layers_mlo)
        self.last_layers = nn.Sequential(*last_layers)

        self.fc = nn.Linear(out_fc * self.fc_expand, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def _forward_implement(self, cc, mlo):
        f_cc = self.early_layers_cc(cc)
        f_mlo = self.early_layers_mlo(mlo)
        if self.aggregation == 'avg':
            x = (f_cc + f_mlo)/2
            # if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
            #     x = self.conv_avg(x)
            #     x = self.bn(x)
            #     x = self.relu(x)
            #     x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # logits = self.softmax(x) 

        if self.aggregation == 'cat':
            x = torch.cat((f_cc,f_mlo),1)
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_ccat(x)
            #     x = self.bn(x)
            #     x = self.relu(x)
            #     x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # logits = self.softmax(x)

        return logits

    def forward(self, cc, mlo):
        logits = self._forward_implement(cc, mlo)
        return logits

class Fusion_VGG(nn.Module):
    def __init__(self, backbone, model_type, aggregation, num_l, num_classes=2, dropout = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.model_type = model_type
        self.fc_expand = 1

        early_layers_cc = []
        early_layers_mlo = []
        mid_layers = []
        last_layers = []

        self.layers = num_l
        child_counter = 0

        if model_type == "PreF":
            for child in backbone.features:
                if child_counter < 0:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim = 3

        if model_type == "EF":
            v = 3*sum(self.layers[:1]) + len(self.layers[:1])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[0].out_channels

        if model_type == "MF":
            v = 3*sum(self.layers[:3]) + len(self.layers[:3])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[-4].out_channels

        if model_type == "LF" or model_type == "PostF" :
            v = 3*sum(self.layers[:5]) + len(self.layers[:5])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[-4].out_channels

        
        self.conv_avg = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False)
        self.conv_ccat = nn.Conv2d(self.out_dim*2 , self.out_dim , kernel_size=1, bias=False)

        if model_type == "PostF":
            early_layers_cc.append(backbone.avgpool)
            early_layers_cc.append(backbone.flatten)
            early_layers_cc.append(backbone.classifier)
            early_layers_mlo.append(backbone.avgpool)
            early_layers_mlo.append(backbone.flatten)
            early_layers_mlo.append(backbone.classifier)
            self.early_layers_l = nn.Sequential(*early_layers_cc)
            self.early_layers_r = nn.Sequential(*early_layers_mlo)
            
            if self.aggregation == 'cat':
                self.fc_expand = 2
            out_fc = self.early_layers[-1][-1].out_features
        else:
            self.early_layers_l = nn.Sequential(*early_layers_cc)
            self.early_layers_r = nn.Sequential(*early_layers_mlo)
            self.mid_layers = nn.Sequential(*mid_layers)

            last_layers.append(self.mid_layers)
            last_layers.append(backbone.avgpool)
            last_layers.append(backbone.flatten)
            last_layers.append(backbone.classifier)
            self.last_layers = nn.Sequential(*last_layers)

            out_fc = self.last_layers[-1][-1].out_features

        self.fc = nn.Linear(out_fc * self.fc_expand, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _forward_implement(self, cc, mlo):
        f_cc = self.early_layers_cc(cc)
        f_mlo = self.early_layers_mlo(mlo)
        if self.aggregation == 'avg':
            x = (f_cc + f_mlo)/2
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_avg(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # logits = self.softmax(x) 

        if self.aggregation == 'cat':
            x = torch.cat((f_cc,f_mlo),1)
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_ccat(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_cc
            x = self.last_layers(x)
            logits = self.fc(x)
            # logits = self.softmax(x)
        
        return logits

    def forward(self, cc, mlo):
        logits = self._forward_implement(cc, mlo)
        return logits


#Fusion for VGG Family  

def fusion_vgg11(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg11_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg13(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg13_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg16(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg16_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg19(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg19_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

#Fusion for ResNet Family

def fusion_resnet18(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet18(pth_url, pretrained), model_type, aggregation, **kwargs)

def fusion_resnet34(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet34(pth_url, pretrained), model_type, aggregation, **kwargs)

def fusion_resnet50(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet50(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnet101(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet101(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnet152(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet152(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext50_32x4d(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext50_32x4d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext101_32x8d(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext101_32x8d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext101_64x4d(pth_url,model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext101_64x4d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_wide_resnet50_2(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(wide_resnet50_2(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_wide_resnet101_2(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(wide_resnet101_2(pth_url, pretrained), model_type, aggregation, **kwargs)








