# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn

from pare.models.backbone import *
from pare.models.backbone.utils import get_backbone_info


class CameraRegressorNetwork(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            num_fc_layers=1,
            num_fc_channels=1024,
            num_out_channels=256,
    ):
        super(CameraRegressorNetwork, self).__init__()
        self.backbone = eval(backbone)(pretrained=True)

        self.num_out_channels = num_out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channels = get_backbone_info(backbone)['n_output_channels']

        assert num_fc_layers > 0, 'Number of FC layers should be more than 0'
        if num_fc_layers == 1:
            self.fc_vfov = nn.Linear(out_channels, num_out_channels)
            self.fc_pitch = nn.Linear(out_channels, num_out_channels)
            self.fc_roll = nn.Linear(out_channels, num_out_channels)

            nn.init.normal_(self.fc_vfov.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_vfov.bias, 0)

            nn.init.normal_(self.fc_pitch.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_pitch.bias, 0)

            nn.init.normal_(self.fc_roll.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_roll.bias, 0)

        else:
            self.fc_vfov = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_pitch = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_roll = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)

    def _get_fc_layers(self, num_layers, num_channels, inp_channels):
        modules = []

        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(inp_channels, num_channels))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_channels, self.num_out_channels))
            else:
                modules.append(nn.Linear(num_channels, num_channels))

        return nn.Sequential(*modules)

    def forward(self, images):
        x = self.backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        vfov = self.fc_vfov(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        return [vfov, pitch, roll]


def test_model():
    backbones = ['resnet50', 'resnet34']
    num_fc_layers = [1, 2, 3]
    num_fc_channels = [256, 512, 1024]
    img_size = [(224, 224), (480,640), (500, 450)]
    from itertools import product

    # print(list(product(backbones, num_fc_layers, num_fc_channels)))
    inp = torch.rand(1, 3, 128, 128)

    for (b, nl, nc, im_size) in list(product(backbones, num_fc_layers, num_fc_channels, img_size)):
        print('backbone', b, 'n_f_layer', nl, 'n_ch', nc, 'im_size', im_size)
        inp = torch.rand(1, 3, *im_size)
        model = CameraRegressorNetwork(backbone=b, num_fc_layers=nl, num_fc_channels=nc)
        out = model(inp)

        breakpoint()
        print('vfov', out[0].shape, 'pitch', out[1].shape, 'roll', out[2].shape)


if __name__ == '__main__':
    test_model()
