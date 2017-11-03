import torch.nn as nn
import torch
import math


class Fire(nn.Module):
    """Implementation of Fire module"""

    def __init__(self, in_channels, out_channels, activation=nn.ReLU,
                 squeeze_ratio=0.5, batchnorm=True, highway=True):
        """Sets up layers for Fire module"""
        super(Fire, self).__init__()

        squeeze_channels = int(in_channels * squeeze_ratio)
        dim_1x1 = int(math.ceil(out_channels / 2))
        dim_3x3 = int(math.floor(out_channels / 2))

        self.highway = highway
        self.batchnorm = batchnorm
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            activation(inplace=True)
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, dim_1x1, kernel_size=1),
            activation(inplace=True)
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, dim_3x3, kernel_size=3, padding=1),
            activation(inplace=True)
        )

    def forward(self, input_data):
        """Forward-propagates data through Fire module"""
        output_data = self.squeeze(input_data)
        output_data = torch.cat([
            self.expand1x1(output_data),
            self.expand3x3(output_data)
        ], 1)
        output_data = output_data + input_data if self.highway else output_data
        output_data = self.norm(output_data) if self.batchnorm else output_data
        return output_data
