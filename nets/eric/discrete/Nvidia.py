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
            nn.Dropout2d(0.5),
            nn.Conv2d(48, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fcl = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 250),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(250, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 200),
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
        x = self.conv_nets(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        x = torch.unbind(x.contiguous().view(x.size(0), 2, 100), 1)
        steering, controls = self.lsm(x[0]), self.lsm2(x[1])
        print(steering.size())
        return steering, controls

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    test_net = Nvidia(5, 6)
    a = test_net(Variable(torch.randn(2, 6*6, 94, 168)),
                 Variable(torch.randn(2, 12, 23, 41)))
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = Nvidia