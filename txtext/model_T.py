import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
import math
from .gradient_layer import cgradient
from .gabor_layer import gabor_layer
from .attention_layer import attention_layer, attention_layer_light

__all__ = ['Conv_BN_ReLU','TANet']

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def upsample(x, size, scale=1):
    _, _, H, W = size
    return F.interpolate(x, size=(int(H // scale), int(W // scale)), mode='bilinear')

class TANet(nn.Module):
    def __init__(self, f_in, C_in, kernel_sizes=[3], theta_size=4):
        super(TANet, self).__init__()
        self.gradient_layer = cgradient(C_in, kernel_size=3)
        self.gabor_layers = nn.Sequential(*[gabor_layer(theta_size, kernel_size=k) for k in kernel_sizes])
        #self.attention_layer = attention_layer(f_in, 1)
        self.attention_layer_light = attention_layer_light(f_in+1+theta_size)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.alphas = ParameterList([Parameter((torch.Tensor([1.0/len(kernel_sizes)]))) for i in range(len(kernel_sizes))])
        self.beta = Parameter(torch.Tensor([1.0]))
        self.kernel_sizes = kernel_sizes
        self.theta_size = theta_size

    def forward(self, feature_map, img):
        img_grad = self.gradient_layer(img) # [batch, 1, H, W]
        img_gabor = [layer(img) for layer in self.gabor_layers] # [batch, theta_size, H, W]
        # weighted sum
        img_gabor[0] = self.alphas[0]*img_gabor[0]
        for i in range(1, len(self.kernel_sizes)):
            img_gabor[0] += self.alphas[i]*img_gabor[i]
        # fusion
        img_fusion = torch.cat((img_gabor[0], self.beta*img_grad), dim=1)

        # downsample layer
        img_fusion = upsample(img_fusion, feature_map.shape)

        # concatenate
        img_fusion = torch.cat((feature_map, img_fusion), dim=1)

        # attention
        feature_map, att_map = self.attention_layer(feature_map, img_fusion)

        return feature_map, att_map

