import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
import torchvision


class read_fun(torchvision.datasets.vision.VisionDataset):
 

    def __init__(self, root, fold, train=True):
        super(read_fun, self).__init__(root, None, None, None)
        assert fold >= 0 and fold <= 3
        self.train = train
        t1_base = os.path.join(root, '')
        voc_base = os.path.join(root, ' ', 'VOC2012')

        t1_train_list_path = os.path.join(root, '', 'train.txt')
        t1_val_list_path = os.path.join(root, '', 'val.txt')
        voc_train_list_path = os.path.join(
            voc_base, '', 'Segmentation', 'train.txt')
        voc_val_list_path = os.path.join(
            voc_base, '', 'Segmentation', 'val.txt')

        t1_train_list = list(np.loadtxt(t1_train_list_path, dtype="str"))
        t1_val_list = list(np.loadtxt(t1_val_list_path, dtype="str"))
        voc_train_list = list(np.loadtxt(voc_train_list_path, dtype="str"))
        voc_val_list = list(np.loadtxt(voc_val_list_path, dtype="str"))

        t1_train_list = t1_train_list + t1_val_list

        t1_train_list = [i for i in t1_train_list if i not in voc_val_list]


        if self.train:
            t1_train_list = [
                i for i in t1_train_list if i not in voc_train_list]

            t1_train_img_list = [os.path.join(
                t1_base, 'img', i + '.jpg') for i in t1_train_list]
            t1_train_target_list = [os.path.join(
                t1_base, 'cls', i + '.mat') for i in t1_train_list]

            voc_train_img_list = [os.path.join(
                voc_base, 'JPEGImages', i + '.jpg') for i in voc_train_list]
            voc_train_target_list = [os.path.join(
                voc_base, "SegmentationClass", i + '.png') for i in voc_train_list]

            # todo:fix datasets and add here
            self.images = t1_train_img_list + voc_train_img_list
            self.targets = t1_train_target_list + voc_train_target_list
        else:

            self.images = [os.path.join(
                voc_base, 'JPEGImages', i + '.jpg') for i in voc_val_list]
            self.targets = [os.path.join(
                voc_base, "SegmentationClass", i + '.png') for i in voc_val_list]

            # self.targets = [os.path.join(
            #     voc_base, "SegmentationClass", i + '.png') for i in voc_val_list]



        self.val_label_set = [i for i in range(fold * 5 + 1, fold * 5 + 6)]
        self.train_label_set = [i for i in range(
            1, 21) if i not in self.val_label_set]
        if self.train:
            self.label_set = self.train_label_set
        else:
            self.label_set = self.val_label_set

        assert len(self.images) == len(self.targets)
        self.to_tensor_func = torchvision.transforms.ToTensor()

       
        folded_images = []
        folded_targets = []

        self.class_img_map = {}
        for label_id, _ in enumerate(self.label_set):
            self.class_img_map[label_id + 1] = []
        self.img_class_map = {}

        for i in range(len(self.images)):
            mask = self.load_seg_mask(self.targets[i])
            appended_flag = False
            for label_id, x in enumerate(self.label_set):
                if x in mask:
                    if not appended_flag:

                        folded_images.append(self.images[i])
                        folded_targets.append(self.targets[i])
                        appended_flag = True
                    cur_img_id = len(folded_images) - 1
                    # cur_class_id = label_id + 1

                    self.class_img_map[cur_class_id].append(cur_img_id)
                    if cur_img_id in self.img_class_map:
                        self.img_class_map[cur_img_id].append(cur_class_id)
                    else:
                        self.img_class_map[cur_img_id] = [cur_class_id]

        self.images = folded_images
        self.targets = folded_targets

    def __len__(self):
        return len(self.images)

    def load_seg_mask(self, file_path):
        if file_path.endswith('.mat'):
            mat = loadmat(file_path)
            target = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
        else:
            target = Image.open(file_path)
        target_np = np.array(target, dtype=np.long)


        target_np[target_np > 20] = 0
        return target_np

    def set_bg_pixel(self, target_np):
        if self.train:
            for x in self.val_label_set:
                target_np[target_np == x] = 0
            max_val_label = max(self.val_label_set)
            target_np[target_np >
                      max_val_label] = target_np[target_np > max_val_label] - 5
        else:
            label_mask_idx_map = []
            for x in self.val_label_set:
                label_mask_idx_map.append(target_np == x)
            target_np = np.zeros_like(target_np)
            for i in range(len(label_mask_idx_map)):
                target_np[label_mask_idx_map[i]] = i + 1
        return target_np

    def get_img_containing_class(self, class_id):
 
        return self.class_img_map[class_id]

    def get_class_in_an_image(self, img_idx):
        return self.img_class_map[img_idx]

    def __getitem__(self, idx):
 
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.to_tensor_func(img)

        target_np = self.load_seg_mask(self.targets[idx])
        target_np = self.set_bg_pixel(target_np)

        return img, torch.tensor(target_np)



# def generalized_imshow(arr):
#     if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
#         arr = arr.permute(1, 2, 0)
#     plt.imshow(arr)
#     plt.show()
