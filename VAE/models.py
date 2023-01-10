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
            nn.Conv2d(1, params['n'], params['kernel_size'], 2),
            nn.ELU(inplace=True),
            nn.Conv2d(params['n'], params['n'] * 2, self.params['kernel_size'], 2),
            nn.ELU(inplace=True),
            nn.Conv2d(params['n'] * 2, params['n'], self.params['kernel_size'], 2),
            nn.ELU(inplace=True),
            nn.Flatten()
        )
        self.var = nn.Linear(params['n'] * 9, params['z'])
        self.mean = nn.Linear(params['n'] * 9, params['z'])

    def forward(self, x):
        main_out = self.main_net(x)
        var = F.relu(self.var(main_out), True)
        mean = self.mean(main_out)
        return var, mean


class DecoderConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.lin = nn.Sequential(
            nn.Linear(params['z'], params['n']*9),
            nn.ELU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(params['n'], params['n'], params['kernel_size'], 2),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(params['n'], params['n'] * 2, params['kernel_size'], 2),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(params['n'] * 2, params['n'], params['kernel_size'], 2, output_padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(params['n'], 1, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z):
        lin = self.lin(z)
        lin = torch.reshape(lin, [-1, self.params['n'], 3, 3])
        return self.net(lin)
