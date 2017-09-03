"""SqueezeNet Supervisor 1.1 modified for regression."""
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
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class PostMetaSqueeze(nn.Module):

    def __init__(self, N_STEPS):
        super(PostMetaSqueeze, self).__init__()

        self.N_STEPS = N_STEPS
        self.post_metadata_features = nn.Sequential(
            Fire(134, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        final_conv = nn.Conv2d(512, self.N_STEPS * 4, kernel_size=1)
        self.post_meta_output = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=6)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.post_metadata_features(x)
        x = self.post_meta_output(x)
        x = x.view(x.size(0), self.N_STEPS, -1)
        return x

class SqueezeNetSupervisor(nn.Module):

    def __init__(self):
        super(SqueezeNetSupervisor, self).__init__()

        self.N_STEPS = 10
        self.metadata_class_count = 6
        self.metadata_size = (11, 20)
        self.pre_metadata_features = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.post_metadata_squeeze = nn.ModuleList([
            PostMetaSqueeze(self.N_STEPS) for _ in range(self.metadata_class_count)
        ])
        self.supervisor_lstm = nn.LSTM(5 * self.metadata_class_count, 4, num_layers=1, batch_first=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, metadata, metadata_index):
        x = self.pre_metadata_features(x)
        x = [torch.cat((x, self.create_metadata(metadata.size(), i)), 1) for i in range(self.metadata_class_count)]
        x = [self.post_metadata_squeeze[i](x[i]) for i in range(self.metadata_class_count)]
        lstm_append = self.create_metadata((x[0].size(0), self.N_STEPS, self.metadata_class_count), metadata_index, 2)
        x = torch.cat(x + [lstm_append], 2)
        x = self.supervisor_lstm(x)[0]
        x = x.contiguous().view(x.size(0), -1)
        return x

    def create_metadata(self, dims, metadata_index, meta_dimension=1):
        pre_meta_dims, post_meta_dims = list(dims[0:meta_dimension]), list(dims[meta_dimension+1:])
        if metadata_index == 0:
            metadata = torch.ones(pre_meta_dims + [1] + post_meta_dims)
        else:
            metadata = torch.zeros(pre_meta_dims + [metadata_index] + post_meta_dims)
            metadata = torch.cat((metadata, torch.ones(pre_meta_dims + [1] + post_meta_dims)), meta_dimension)

        if metadata_index != self.metadata_class_count - 1:
            metadata = torch.cat((metadata, torch.zeros(pre_meta_dims + [dims[meta_dimension] - metadata_index - 1] + post_meta_dims)), meta_dimension)
        return metadata


def unit_test():
    test_net = SqueezeNetSupervisor()
    # print test_net.create_metadata(Variable(torch.randn(5, 6, 11, 20)), 0)
    a = test_net(Variable(torch.randn(5, 14, 94, 168)),
                 Variable(torch.randn(5, 6, 11, 20)), 1)
    logging.debug('Net Test Output = {}'.format(a))
    logging.debug('Network was Unit Tested')


unit_test()
