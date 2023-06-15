# Hailo #

import torch.nn as nn
import numpy as np
import torch
from ..backbones.repvgg import RepVGGBlock


class ConvBNAct(nn.Module):  # TODO: Amit -- can replace with `BaseConv` by YOLOX
    '''Conv2d + BN + Activation (ReLU)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Transpose(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample_transpose = torch.nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # self.upsample_transpose = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample_transpose(x)


def make_repvgg_stage(in_channels, out_channels, num_blocks, stride=1, deploy=False):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    for stride in strides:
        blocks.append(RepVGGBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=stride, padding=1, groups=1, deploy=deploy, use_se=False))
        in_channels = out_channels
    return nn.Sequential(*blocks)


class PANRepVGGNeck(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        num_repeats=None,
        bb_channels_list=None,
        neck_channels_list=None
    ):
        super().__init__()
        assert num_repeats is not None
        assert bb_channels_list is not None
        assert neck_channels_list is not None

        neck_num_repeats = [int(round(r * depth)) for r in num_repeats]  # [3, 4, 4, 3]
        channels = [int(round(ch * width)) for ch in bb_channels_list + neck_channels_list]

        self.neck_p4 = make_repvgg_stage(channels[2] + channels[4], channels[4], neck_num_repeats[0])  # 192 --> 64
        self.neck_p3 = make_repvgg_stage(channels[1] + channels[5], channels[5], neck_num_repeats[1])  # 96 --> 32

        self.align_channels0 = ConvBNAct(in_channels=channels[3], out_channels=channels[4], kernel_size=1)  # 256 --> 64
        self.upsample0 = Transpose()
        self.align_channels1 = ConvBNAct(in_channels=channels[4], out_channels=channels[5], kernel_size=1)  # 64 --> 32
        self.upsample1 = Transpose()

    def forward(self, input):
        """
        Args:
            input (Tuple[Tensor]): backbone output features:
                outputs[0] - /8 resolution feature map
                outputs[1] - /16 resolution feature map
                outputs[2] - /32 resolution feature map
        Returns:
            List[Tensor]: neck output features:
                outputs[0] - /8 resolution feature map
        """
        (x2, x1, x0) = input

        fpn_out0 = self.align_channels0(x0)
        upsample_features0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_features0, x1], 1)
        f_out0 = self.neck_p4(f_concat_layer0)

        fpn_out1 = self.align_channels1(f_out0)
        upsample_features1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_features1, x2], 1)
        pan_out2 = self.neck_p3(f_concat_layer1)

        return pan_out2