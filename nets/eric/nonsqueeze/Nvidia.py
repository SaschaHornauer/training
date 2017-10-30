"""SqueezeNet 1.1 modified for regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)

class Nvidia(nn.Module):

    def __init__(self, n_steps=10, n_frames=2):
        super(Nvidia, self).__init__()

        self.n_steps = n_steps
        self.n_frames = n_frames
        self.conv_nets = nn.Sequential(
            nn.Conv2d(3 * 2 * self.n_frames, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(48, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fcl = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 100),
            nn.Linear(100, 2 * self.n_steps)
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                else:
                    init.xavier_normal(mod.weight.data)
            if hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
                init.normal(mod.bias.data, 0, 0.1)


    def forward(self, x, metadata):
        x = self.conv_nets(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        x = x.contiguous().view(x.size(0), -1, 2)
        return x

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    test_net = Nvidia(1, 30)
    a = test_net(Variable(torch.randn(1, 30*6, 94, 168)),
                 Variable(torch.randn(1, 12, 23, 41)))
    sizes = [1, 1, 2]
    assert(all(a.size(i) == sizes[i] for i in range(len(sizes))))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = Nvidia