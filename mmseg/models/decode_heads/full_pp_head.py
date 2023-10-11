# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ..builder import HEADS
from .decode_head import BaseDecodeHead



@HEADS.register_module()
class PostProcess(BaseDecodeHead):
    def __init__(self, num_convs,  **kwargs):
        super(PostProcess, self).__init__(**kwargs)

        self.num_convs = num_convs
        self.output_channels = self.channels
        self.conv_layers = nn.ModuleList()
        in_channels = self.in_channels
        self.num_classes = kwargs['num_classes']
        for i in range(self.num_convs):
            self.conv_layers.append(nn.Conv2d(in_channels, self.output_channels, kernel_size=3, padding=1))
            in_channels = self.output_channels

        self.dw = torch.nn.Conv2d(in_channels=self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=(1, 2), groups=self.num_classes - 1, bias=False)
        w = np.ones((self.num_classes - 1,1,1,2), dtype=np.float32)
        # w[:, 0, 0, 0] = -1
        w[:, :, :, 1] = -1
        self.dw.weight = torch.nn.Parameter(torch.Tensor(w))
        self.relu = torch.nn.ReLU()
        self.argmax_star = True  #  Argmax* (Argmax from bottom) -> will not compile, so default is False

    def forward(self, x):

        # channels in -> channels out (several, default is 1 convs layers after backbone)
        for conv_layer in self.conv_layers:
            x = nn.functional.relu(conv_layer(x))
        
        # #channels -> #classes
        x  = self.cls_seg(x)

        # input is (BxCxHxW): 1x10x92x120 output is 1x10x736x240
        x = torch.nn.functional.interpolate(x, size=(736, 240), mode='bilinear', align_corners=True)

        # argmax on channels. output is 1x1x736x240
        x = torch.argmax(x, dim=1, keepdim=True)

        # H<->W transpose. output is 1x1x240x736
        x = torch.transpose(x, 2, 3)

        # torch.nn.functional.one_hot adds an extra dim at the end of the tensor so output is 1x1x240x736x10
        x = torch.nn.functional.one_hot(x, num_classes=self.num_classes)  
        x = torch.transpose(x, 1, 4)  # output is 1x10x240x736x1
        x = torch.squeeze(x, dim=-1)  # output is 1x10x240x736

        # First output edge detector
        out1 = x[:, :-1, :, :]  # output is 1x9x240x736. Assuming raindrop is last class
        # out1 = torch.nn.functional.pad(out1, [1, 0, 0, 0])  # output is 1x9x240x737
        out1 = out1.to(torch.float32)
        out1 = torch.nn.functional.pad(out1, [0, 1, 0, 0], mode='constant', value=0.5)  # output is 1x9x240x737
        out1 = self.relu(self.dw(out1))  # output is 1x9x240x736

        # W<->C transpose. output is 1x736x240x9
        out1 = torch.transpose(out1, 1, 3)

        if self.argmax_star:
            # argmax* support: Flip the 736 axis. output is 1x736x240x9
            out1 = torch.flip(out1, dims=(1,))
        # argmax on channels. final output is 1x1x240x9
        out1 = torch.argmax(out1, dim=1, keepdim=True)

        # second output is to reduce sum on final class. output is 4 integers of 1x1x1x1 so each one would be represented by 16 bit integer. Assuming raindrop is last class
        out2, out3, out4, out5 = x[:, -1:, :, :184], x[:, -1:, :, 184:368], x[:, -1:, :, 368:552], x[:, -1:, :, 552:]
        out2 = torch.sum(torch.sum(out2, dim=-1, keepdim=True), dim=-2, keepdim=True)
        out3 = torch.sum(torch.sum(out3, dim=-1, keepdim=True), dim=-2, keepdim=True)
        out4 = torch.sum(torch.sum(out4, dim=-1, keepdim=True), dim=-2, keepdim=True)
        out5 = torch.sum(torch.sum(out5, dim=-1, keepdim=True), dim=-2, keepdim=True)

        return out1, out2, out3, out4, out5
