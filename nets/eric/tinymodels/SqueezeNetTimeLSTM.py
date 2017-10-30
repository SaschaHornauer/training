"""SqueezeNet 1.1 modified for LSTM regression."""
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import random
from torch.autograd import Variable

logging.basicConfig(filename='training.log', level=logging.DEBUG)


# from Parameters import ARGS


class Fire(nn.Module):  # pylint: disable=too-few-public-methods
    """Implementation of Fire module"""

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        """Sets up layers for Fire module"""
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

    def forward(self, input_data):
        """Forward-propagates data through Fire module"""
        output_data = self.squeeze_activation(self.squeeze(input_data))
        return self.norm(torch.cat([
            self.expand1x1_activation(self.expand1x1(output_data)),
            self.expand3x3_activation(self.expand3x3(output_data))
        ], 1))


class SqueezeNetTimeLSTM(nn.Module):  # pylint: disable=too-few-public-methods
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self, n_frames=2, n_steps=10):
        """Sets up layers"""
        super(SqueezeNetTimeLSTM, self).__init__()

        self.is_cuda = False
        self.requires_controls = True

        self.n_frames = n_frames
        self.n_steps = n_steps
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(3 * 2, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(16, 4, 8, 8),
        )
        self.post_metadata_features = nn.Sequential(
            Fire(16, 6, 12, 12),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(24, 8, 16, 16),
            Fire(32, 8, 16, 16),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 12, 24, 24),
            Fire(48, 12, 24, 24),
            Fire(48, 16, 32, 32),
            Fire(64, 16, 32, 32),
        )
        final_conv = nn.Conv2d(64, 8, kernel_size=1)
        self.pre_lstm_output = nn.Sequential(
            nn.Dropout(0.75),
            final_conv,
            nn.AvgPool2d(kernel_size=5, stride=5),
        )
        self.lstm_encoder = nn.ModuleList([
            nn.LSTM(16, 32, 1, batch_first=True)
        ])
        self.lstm_decoder = nn.ModuleList([
            nn.LSTM(2, 32, 1, batch_first=True)
        ])
        self.output_linear = nn.Sequential(nn.Linear(32, 2))

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
            if hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
                init.normal(mod.bias.data, 0, 0.0001)



    def forward(self, camera_data, metadata, controls=None):
        """Forward-propagates data through SqueezeNetTimeLSTM"""
        batch_size = camera_data.size(0)
        metadata = metadata.contiguous().view(-1, 8, 23, 41)
        net_output = camera_data.contiguous().view(-1, 6, 94, 168)
        net_output = self.pre_metadata_features(net_output)
        # net_output = torch.cat((net_output, metadata), 1)
        net_output = self.post_metadata_features(net_output)
        net_output = self.pre_lstm_output(net_output)
        net_output = net_output.contiguous().view(batch_size, -1, 16)
        for lstm in self.lstm_encoder:
            net_output, last_hidden_cell = lstm(net_output)
            last_hidden_cell = list(last_hidden_cell)
        random.seed(None)
        should_generate = random.random < 0.5
        print(should_generate)
        if (controls is not None) and (not should_generate):
            for lstm in self.lstm_decoder:
                if last_hidden_cell:
                    # last_hidden_cell[0] = last_hidden_cell[0].contiguous().view(batch_size, -1, 256)
                    # last_hidden_cell[1] = last_hidden_cell[1].contiguous().view(batch_size, -1, 256)
                    net_output = lstm(self.get_decoder_seq(controls), last_hidden_cell)[0]
                    last_hidden_cell = None
                else:
                    net_output = lstm(net_output)[0]
            net_output = self.output_linear(net_output.contiguous().view(-1, 32))
        else:
            print 'generating'
            net_output = Variable(torch.zeros(batch_size, self.n_steps, 2))
            net_output = net_output.cuda() if self.is_cuda else net_output
            for lstm in self.lstm_decoder:
                for i in range(self.n_steps):
                    if i == 0:
                        init_input = Variable(torch.zeros(batch_size, 1, 2))
                        init_input = init_input.cuda() if self.is_cuda else init_input
                        lstm_output, last_hidden_cell = lstm(init_input, last_hidden_cell)
                    else:
                        lstm_output, last_hidden_cell = lstm(net_output[:,i-1,:].unsqueeze(1), last_hidden_cell)
                    linear = self.output_linear(lstm_output.contiguous().view(-1, 32))
                    net_output[:, i, :] = linear
        net_output = net_output.contiguous().view(batch_size, -1, 2)
        return net_output

    def get_decoder_seq(self, controls):
        controls = controls.clone()
        controls[:,1:,:] = controls[:,0:controls.size(1)-1,:]
        controls[:,0,:] = 0
        decoder_input_seq = Variable(controls)
        return decoder_input_seq.cuda() if self.is_cuda else decoder_input_seq


    def cuda(self, device_id=None):
        self.is_cuda = True
        return super(SqueezeNetTimeLSTM, self).cuda(device_id)

    def num_params(self):
        return sum([reduce(lambda x, y: x * y, [dim for dim in p.size()], 1) for p in self.parameters()])

def unit_test():
    """Tests SqueezeNetTimeLSTM for size constitency"""
    test_net = SqueezeNetTimeLSTM(6, 10)
    test_net_output = test_net(
        Variable(torch.randn(1, 36, 94, 168)),
        Variable(torch.randn(1, 6, 8, 23, 41))
        # torch.randn(1, 10, 2)
    )
    sizes = [1, 10, 2]
    assert(all(test_net_output.size(i) == sizes[i] for i in range(len(sizes))))
    logging.debug('Net Test Output = {}'.format(test_net_output))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = SqueezeNetTimeLSTM