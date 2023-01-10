import torch
from torch import nn
import torch.nn.functional as F


class EncoderLinear(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.main_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, params['n'] * 2),
            nn.ELU(inplace=True),
            nn.Linear(params['n'] * 2, params['n'] * 1),
            nn.ELU(inplace=True),
        )
        self.var = nn.Linear(params['n'], params['z'])
        self.mean = nn.Linear(params['n'], params['z'])

    def forward(self, x):
        main_out = self.main_net(x)
        var = F.relu(self.var(main_out), True)
        mean = self.mean(main_out)
        return var, mean


class DecoderLinear(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.net = nn.Sequential(
            nn.Linear(params['z'], params['n']),
            nn.ELU(inplace=True),
            nn.Linear(params['n'], params['n'] * 2),
            nn.ELU(inplace=True),
            nn.Linear(params['n'] * 2, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class EncoderConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.main_net = nn.Sequential(
            nn.Conv2d(1, self.params['n'], self.params['kernel_size'], padding='same', bias=False),
            nn.ELU(inplace=True),  # n * 32 * 32
            self.conv_block(1, 2),  # 2n * 16 * 16
            self.conv_block(2, 4),  # 4n * 8 * 8
            self.conv_block(4, 4),  # 4n * 4 * 4
            nn.Conv2d(params['n'] * 4, params['n'], params['kernel_size'], padding='same', bias=False),
            nn.ELU(inplace=False),
        )
        self.var = nn.Conv2d(params['n'], params['z'], params['kernel_size'], padding='same', bias=False)
        self.mean = nn.Conv2d(params['n'], params['z'], params['kernel_size'], padding='same', bias=False)

    def conv_block(self, in_mult, out_mult):
        return nn.Sequential(
            nn.Conv2d(self.params['n'] * in_mult, self.params['n'] * out_mult,
                      self.params['kernel_size'], padding='same', bias=False),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        main_out = self.main_net(x)
        var = F.relu(self.var(main_out), True)
        mean = self.mean(main_out)
        return var, mean


class DecoderConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.net = nn.Sequential(
            nn.Conv2d(params['z'], params['n'] * 4, params['kernel_size'], padding='same', bias=False),
            nn.ELU(inplace=False),  # 4n * 4 * 4

            self.conv_block(4, 4),  # 4n * 8 * 8
            self.conv_block(4, 2),  # 2n * 16 * 16
            self.conv_block(2, 1),  # n * 32 * 32
            nn.Conv2d(params['n'], 1, params['kernel_size'], padding='same', bias=False),
            nn.Sigmoid(),
        )

    def conv_block(self, in_mult, out_mult):
        return nn.Sequential(
            nn.Conv2d(self.params['n'] * in_mult, self.params['n'] * out_mult,
                      self.params['kernel_size'], padding='same', bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

    def forward(self, z):
        return self.net(z)
