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
        super(Fire, self).__init__()
        self.norm = torch.nn.BatchNorm2d(expand1x1_planes + expand3x3_planes)
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.norm(torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1))


class SqueezeNet(nn.Module):

    def __init__(self, n_steps=10, n_frames=2):
        super(SqueezeNet, self).__init__()

        self.n_steps = n_steps
        self.n_frames = n_frames
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(3 * 2 * self.n_frames, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            Fire(16, 4, 8, 8)
        )
        self.post_metadata_features = nn.Sequential(
            Fire(16, 6, 12, 12),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(24, 8, 16, 16),
            Fire(32, 8, 16, 16),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 12, 24, 24),
            Fire(48, 12, 24, 24),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(48, 16, 32, 32),
            nn.Dropout2d(p=0.5),
            Fire(64, 16, 32, 32)
        )
        final_conv = nn.Conv2d(64, self.n_steps, kernel_size=1)
        self.final_output = nn.Sequential(
            nn.Dropout2d(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=5),
            nn.Sigmoid()
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
            if hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
                init.normal(mod.bias.data, 0, 0.0001)

    def forward(self, x, metadata):
        x = self.pre_metadata_features(x)
        # x = torch.cat((x, metadata), 1)
        x = self.post_metadata_features(x)
        x = self.final_output(x)
        x = x.view(x.size(0), -1, 2)
        return x

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    test_net = SqueezeNet(1, 6)
    a = test_net(Variable(torch.randn(1, 6*6, 94, 168)),
                 Variable(torch.randn(1, 8, 23, 41)))
    sizes = [1, 1, 2]
    assert(all(a.size(i) == sizes[i] for i in range(len(sizes))))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = SqueezeNet