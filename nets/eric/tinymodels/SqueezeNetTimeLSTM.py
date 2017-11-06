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
        self.final_output = nn.Sequential(
            torch.nn.BatchNorm2d(expand1x1_planes + expand3x3_planes),
            nn.Dropout2d(p=0.2)
        )
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ELU(inplace=True)
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



class SqueezeNetTimeLSTM(nn.Module):  # pylint: disable=too-few-public-methods
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self, n_frames=2, n_steps=10):
        """Sets up layers"""
        super(SqueezeNetTimeLSTM, self).__init__()

        self.is_cuda = False
        self.requires_controls = True

        self.n_frames = n_frames
        self.n_steps = n_steps
        self.pre_lstm_output = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Dropout2d(p=0.2),

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

            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(16),
        )
        self.lstm_encoder = nn.ModuleList([
            nn.LSTM(32, 64, 1, batch_first=True)
        ])
        self.lstm_decoder = nn.ModuleList([
            nn.LSTM(32, 64, 1, batch_first=True)
        ])
        self.post_lstm_linear = nn.Sequential(
                                            nn.Linear(64, 32),
                                            nn.ELU(inplace=True),
                                            nn.BatchNorm1d(32),
                                              )
        self.output_linear = nn.Sequential(
                                            # nn.Dropout(p=.5),
                                            nn.Linear(32, 16),
                                            nn.ELU(inplace=True),
                                            nn.BatchNorm1d(16),
                                            nn.Linear(16, 2),
                                            nn.Sigmoid()
                                           )

        for mod in self.pre_lstm_output.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
                else:
                    init.normal(mod.weight.data)
            # elif hasattr(mod, 'bias') and hasattr(mod.bias, 'data'):
            #     init.normal(mod.bias.data, mean=0, std=0.000000001)


    def forward(self, camera_data, metadata, controls=None):
        """Forward-propagates data through SqueezeNetTimeLSTM"""
        batch_size = camera_data.size(0)
        net_output = camera_data.contiguous().view(-1, 6, 94, 168)
        net_output = self.pre_lstm_output(net_output)
        net_output = net_output.contiguous().view(batch_size, -1, 32)
        for lstm in self.lstm_encoder:
            net_output, last_hidden_cell = lstm(net_output)
            # last_hidden_cell = list(last_hidden_cell)
        # for lstm in   self.lstm_decoder:
        #     if last_hidden_cell:
        #         net_output = lstm(self.get_decoder_input(camera_data), last_hidden_cell)[0]
        #         last_hidden_cell = None
        #     else:
        #         net_output = lstm(net_output)[0]

        # Initialize the decoder sequence
        init_input = Variable(torch.ones(batch_size, 1, 32) * 0.5)
        init_input = init_input.cuda() if self.is_cuda else init_input
        lstm_output, last_hidden_cell = self.lstm_decoder[0](init_input, last_hidden_cell)
        init_input = self.post_lstm_linear(lstm_output.contiguous().squeeze(1)).unsqueeze(1)

        if (controls is not None): #and (not self.is_generating):
            for lstm in self.lstm_decoder:
                if last_hidden_cell:
                    net_output = lstm(self.get_decoder_seq(controls), last_hidden_cell)[0]
                    last_hidden_cell = None
                else:
                    net_output = lstm(net_output)[0]
            # net_output = last_hidden_cell[0]
            net_output = self.output_linear(self.post_lstm_linear(net_output.contiguous().view(-1, 64)))
        else:
            list_outputs = []
            list_lstm_inputs = []
            for lstm in self.lstm_decoder:
                for i in range(self.n_steps):
                    if i == 0:
                        lstm_output, last_hidden_cell = lstm(init_input, last_hidden_cell)
                    else:
                        lstm_output, last_hidden_cell = lstm(list_lstm_inputs[i-1], last_hidden_cell)
                    linear = self.post_lstm_linear(lstm_output.contiguous().view(-1, 64))
                    list_lstm_inputs.append(linear.unsqueeze(1))
                    linear = self.output_linear(linear)
                    list_outputs.append(linear.unsqueeze(1))
            net_output = torch.cat(list_outputs, 1)
        # net_output = self.output_linear(net_output.contiguous().view(-1, 64))
        net_output = net_output.contiguous().view(batch_size, -1, 2)
        return net_output

    def get_decoder_input(self, camera_data):
        batch_size = camera_data.size(0)
        input = Variable(torch.zeros(batch_size, self.n_steps, 2))
        return input.cuda() if self.is_cuda else input


    def get_decoder_seq(self, controls):
        controls = controls.clone()
        if controls.size(1) > 1:
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
    test_net = SqueezeNetTimeLSTM(6, 5)
    test_net_output = test_net(
        Variable(torch.randn(2, 6 * 6, 94, 168)),
        Variable(torch.randn(2, 6, 8, 23, 41))
    )
    sizes = [2, 5, 2]
    print test_net_output.size()
    assert(all(test_net_output.size(i) == sizes[i] for i in range(len(sizes))))
    logging.debug('Net Test Output = {}'.format(test_net_output))
    logging.debug('Network was Unit Tested')
    print(test_net.num_params())

unit_test()

Net = SqueezeNetTimeLSTM