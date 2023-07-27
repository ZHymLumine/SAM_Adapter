import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class VisionTransformerUpHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=64, embed_dim=1024, mid_channels=256,
                 norm_cfg=None, num_conv=4, upsampling_method='bilinear', num_upsampe_layer=4, conv3x3_conv1x1=True, align_corners=False, **kwargs):
        super(VisionTransformerUpHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer
        self.conv3x3_conv1x1 = conv3x3_conv1x1
        self.align_corners = align_corners

        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Sequential(
                    nn.Conv2d(embed_dim, mid_channels, kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(mid_channels)
                    )
            else:
                self.conv_0 = nn.Conv2d(embed_dim, 256, 1, 1)
            self.conv_1 = nn.Conv2d(256, 1, 1, 1)

        elif self.num_conv == 4:
            self.conv_0 = nn.Sequential(nn.Conv2d(
                embed_dim, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True))
            self.conv_1 = nn.Sequential(nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True))
            self.conv_2 = nn.Sequential(nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True))
            self.conv_3 = nn.Sequential(nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True))
            self.conv_4 = nn.Sequential(nn.Conv2d(
                mid_channels, 1, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(mid_channels))
            

        # Segmentation head

    def forward(self, inputs):
        x = torch.cat([inputs[0], inputs[1], inputs[2], inputs[3]], dim=1)

        if self.upsampling_method == 'bilinear':
            # if x.dim() == 3:
            #     n, hw, c = x.shape
            #     h = w = int(math.sqrt(hw))
            #     x = x.transpose(1, 2).reshape(n, c, h, w)

            if self.num_conv == 2:
                if self.num_upsampe_layer == 2:
                    x = self.conv_0(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
                elif self.num_upsampe_layer == 1:
                    x = self.conv_0(x)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_conv == 4:
                if self.num_upsampe_layer == 4:
                    x = self.conv_0(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

                    x = self.conv_2(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    
                    x = self.conv_3(x)

                    x = self.conv_4(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                   
        return x