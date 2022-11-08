#!/usr/bin/python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchsummary

###  ResNet Setting   ###

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is applied.")

class conv_block(nn.Module):
    def __init__(self, in_depth, out_depth, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False, **kwargs)
        self.batch = nn.BatchNorm2d(out_depth)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv(x)
        output = self.batch(output)
        output = self.relu(output)
        return output

class residual_block(nn.Module):
    def __init__(self, in_depth, out_depth, stride=1, **kwargs):
        super(residual_block, self).__init__()
        self.stride= stride
        self.conv1 = conv_block(in_depth, out_depth, stride=stride)
        self.conv2 = nn.Conv2d(out_depth, out_depth, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.batch = nn.BatchNorm2d(out_depth)

        if stride == 2:
            self.dim_reduc_conv= nn.Conv2d(in_depth, out_depth, kernel_size=1, stride=stride, bias=False, **kwargs)
            self.dim_reduc_batch = nn.BatchNorm2d(out_depth)

    def forward(self, x):
        output = self.conv1(x)
        output = self.batch(self.conv2(output))
        if self.stride == 2:
            output += self.dim_reduc_batch(self.dim_reduc_conv(x))
        else:
            output += x
        output = F.relu(output)
        return output

class get_layers(nn.Module):
    def __init__(self, in_depth, out_depth, stride, num_layers):
        super(get_layers, self).__init__()
        count = 0
        self.layers = nn.Sequential(residual_block(in_depth, out_depth, stride=stride))
        while count < num_layers-1:
            self.layers.append(residual_block(out_depth, out_depth, stride=1))
            count += 1

    def forward(self, x):
        output = self.layers(x)
        return output

class ResNet(nn.Module):
    def __init__(self, num_layers):
        super(ResNet, self).__init__()
        self.conv1 = conv_block(3, 16)

        self.layers_16 = get_layers(16,16, stride=1, num_layers=num_layers)
        self.layers_32 = get_layers(16,32, stride=2, num_layers=num_layers)
        self.layers_64 = get_layers(32,64, stride=2, num_layers=num_layers)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.conv1(x)

        output = self.layers_16(output)
        output = self.layers_32(output)
        output = self.layers_64(output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

resnet = ResNet(3).to(device)
torchsummary.summary(resnet, (3,32,32))
