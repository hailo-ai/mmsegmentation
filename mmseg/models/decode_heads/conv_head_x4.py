# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ConvHeadx4(BaseDecodeHead):
    def __init__(self,
                num_convs=2,
                **kwargs):
        super(ConvHeadx4, self).__init__(**kwargs)

        self.num_convs = num_convs
        self.output_channels = self.channels

        self.bilinear_resize = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_y = nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=True)

        self.conv_layers = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.num_convs):
            self.conv_layers.append(nn.Conv2d(in_channels, self.output_channels, kernel_size=3, padding=1))
            in_channels = self.output_channels


    def forward(self, inputs):
        """Forward function."""
        x = inputs[self.in_index]
        x = self.bilinear_resize(x)   # [h/8, w/8] -> [h/4, w/4]

        for conv_layer in self.conv_layers:
            x = nn.functional.relu(conv_layer(x))

        x  = self.cls_seg(x)
        output = self.upsample_y(x)  # [h/4, w/4] -> [h, w/4]

        return output