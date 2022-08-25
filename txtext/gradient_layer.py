
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

__all__ = ['cgradient']

class cgradient(nn.Module):
    def __init__(self, in_planes=3, kernel_size=3):
        super(cgradient, self).__init__()
        self.eps = 10**(-12)
        self.in_planes = in_planes
        self.sobel_x = torch.Tensor([[-3.0, 0.0, 3.0],[-10.0, 0.0, 10.0],[-3.0, 0.0, 3.0]])
        self.sobel_y = torch.Tensor([[-3.0, -10.0, -3.0],[0.0, 0.0, 0.0],[3.0, 10.0, 3.0]])
        self.weight_x = Parameter(self.sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_planes,1,1,1), requires_grad=False)
        self.weight_y = Parameter(self.sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_planes,1,1,1), requires_grad=False)
        self.conv2d = F.conv2d

    def forward(self, imgs):
        gxx = self.conv2d(imgs, self.weight_x, stride=1, padding=1, groups=self.in_planes)
        gxx = torch.sum(gxx**2, axis=1)
        gyy = self.conv2d(imgs, self.weight_y, stride=1, padding=1, groups=self.in_planes)
        gyy = torch.sum(gyy**2, axis=1)
        gxy = self.conv2d(imgs, self.weight_x, stride=1, padding=1, groups=self.in_planes)*self.conv2d(imgs, self.weight_y, stride=1, padding=1, groups=self.in_planes)
        gxy = torch.sum(gxy, axis=1)
        
        theta = 0.5*torch.atan(torch.div(2*gxx, gxx-gyy+self.eps)) # color gradient orientation
        G = (gxx + gyy) + (gxx - gyy)*torch.cos(2*theta) + 2*gxy*torch.sin(2*theta) # color gradient magnitude
        G[G < 0] = 0
        f = torch.sqrt(0.5*G)

        return f.unsqueeze(1)

