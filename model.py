'''
用于存放模型结构
'''
from torchvision import models
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import config

class DeepLabV3Res101(nn.Module):
    def __init__(self):
        super(DeepLabV3Res101, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=config.num_classes)

    def forward(self, x:Tensor)->Tensor:
        return self.model(x)['out']

class DeepLabV3Res50(nn.Module):
    def __init__(self):
        super(DeepLabV3Res50, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=config.num_classes)

    def forward(self, x:Tensor)->Tensor:
        return self.model(x)['out']