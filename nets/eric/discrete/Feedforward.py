"""SqueezeNet 1.1 modified for regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)

class Feedforward(nn.Module):

    def __init__(self, n_steps=10, n_frames=2):
        super(Feedforward, self).__init__()

        self.n_steps = n_steps
        self.n_frames = n_frames
        self.final_output = nn.Sequential(
            nn.Conv2d(6 * self.n_frames, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(64, 82, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(82),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(82, 100, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(100),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1),
        )
        self.lsm = nn.LogSoftmax()
        self.lsm2 = nn.LogSoftmax()

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
                else:
                    init.normal(mod.weight.data)
            if hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
                init.normal(mod.bias.data, 0, 0.01)

    def forward(self, x, metadata):
        x = torch.unbind(self.final_output(x).squeeze(2), 2)
        steering, controls = self.lsm(x[0]), self.lsm2(x[1])
        return steering, controls

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    test_net = Feedforward(5, 6)
    a = test_net(Variable(torch.randn(2, 6*6, 94, 168)),
                 Variable(torch.randn(2, 8, 23, 41)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = Feedforward