"""SqueezeNet 1.1 modified for regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        """Sets up layers for Fire module"""
        super(Fire, self).__init__()
        self.final_output = nn.Sequential(
            torch.nn.BatchNorm2d(expand1x1_planes + expand3x3_planes)
        )
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.should_iterate = inplanes == (expand3x3_planes + expand1x1_planes)

    def forward(self, input_data):
        """Forward-propagates data through Fire module"""
        output_data = self.squeeze_activation(self.squeeze(input_data))
        output_data = torch.cat([
            self.expand1x1_activation(self.expand1x1(output_data)),
            self.expand3x3_activation(self.expand3x3(output_data))
        ], 1)
        output_data = output_data + input_data if self.should_iterate else output_data
        output_data = self.final_output(output_data)
        return output_data



class SqueezeNet(nn.Module):

    def __init__(self, n_steps=10, n_frames=2):
        super(SqueezeNet, self).__init__()

        self.n_steps = n_steps
        self.n_frames = n_frames
        self.final_output = nn.Sequential(
            nn.Conv2d(6 * self.n_frames, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),

            Fire(16, 4, 8, 8),
            Fire(16, 12, 12, 12),
            Fire(24, 16, 16, 16),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 16, 16, 16),
            Fire(32, 24, 24, 24),
            nn.Dropout2d(p=0.5),
            Fire(48, 24, 24, 24),
            Fire(48, 32, 32, 32),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 32, 32, 32),
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(32, self.n_steps, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Sigmoid(),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.xavier_normal(mod.weight.data, gain=1.5)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
                else:
                    init.normal(mod.weight.data)
            if hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
                init.normal(mod.bias.data, 0.01)

    def forward(self, x, metadata):
        x = self.final_output(x)
        x = x.view(x.size(0), -1, 2)
        return x

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    test_net = SqueezeNet(5, 6)
    a = test_net(Variable(torch.randn(2, 6*6, 94, 168)),
                 Variable(torch.randn(2, 8, 23, 41)))
    sizes = [2, 5, 2]
    assert(all(a.size(i) == sizes[i] for i in range(len(sizes))))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = SqueezeNet