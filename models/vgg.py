"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)

def norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)

I = cv2.imread('pebbles.jpg')
I2 = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
gray2 = np.copy(gray.astype(np.float64))
(rows, cols) = gray.shape[:2]

conv_maps = np.zeros((rows, cols,16),np.float64)

filter_vectors = np.array([[1, 4, 6,  4, 1],
                            [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])

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

