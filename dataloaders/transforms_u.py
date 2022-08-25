"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        scribble = tr_F.resize(scribble, self.size, interpolation=Image.ANTIALIAS)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class DilateScribble(object):
    """
    Dilate the scribble mask

    Args:
        size: window width
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        scribble = sample['scribble']
        dilated_scribble = Image.fromarray(
            ndimage.minimum_filter(np.array(scribble), size=self.size))
        dilated_scribble.putpalette(scribble.getpalette())

        sample['scribble'] = dilated_scribble
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
        inst = torch.Tensor(np.array(inst)).long()
        scribble = torch.Tensor(np.array(scribble)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble


            filters = list()
        for ii in range(4):
            for jj in range(4):
                filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))

        smooth_kernel = (1/25)*np.ones((5,5))
        gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
        gray_processed = np.abs(gray2 - gray_smooth)

        for ii in range(len(filters)):
            conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

        texture_maps = list()
        texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
        texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
        texture_maps.append(norm(conv_maps[:, :, 10]))
        texture_maps.append(norm(conv_maps[:, :, 15]))
        texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
        texture_maps.append(norm(conv_maps[:, :, 5]))
        texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
        texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))
        return sample
