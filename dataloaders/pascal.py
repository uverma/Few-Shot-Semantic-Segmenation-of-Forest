"""
Load pascal VOC dataset
"""

import os

import numpy as np
from PIL import Image
import torch

from .common import BaseDataset
palette = { #black
           (0,255,0) : 150, #green
           (0,0,255) : 10, #blue
           (255,0,255) : 80, #pink
           (0,255,255) : 100, #light blue
           (255,255,0): 40 #white

          }

# palette = {(0,   0,   0) : 0 , # black
#            (0,  0, 255) : 1 , # blue
#            (255,  0,  0) : 2, 
#            (255,255,  255) : 3, #white
#            (  0,255,  0) : 4, #green
#            (255,  0,255) : 5, # pink
#            (  0,255,255) : 6, 
#            (255,  0,153) : 7,
#            (153,  0,255) : 8,
#            (  0,153,255) : 9,
#            (153,255,  0) : 10,
#            (255,153,  0) : 11,
#            (  0,255,153) : 12,
#            (  0,153,153) : 13
#           }


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')
        self._id_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.transforms = transforms
        self.to_tensor = to_tensor

        with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        
        semantic = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        semantic = np.array(semantic)
        seman2=Image.fromarray(convert_from_color_segmentation(semantic))
        semantic_mask=seman2
        # a=np.asarray(semantic_mask)
        # print(a.shape)
        # semantic_mask.save('test.png')
        # exit()
        # print(len(image.split()))
        instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        scribble_mask = Image.open(os.path.join(self._scribble_dir, f'{id_}.png'))
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample









