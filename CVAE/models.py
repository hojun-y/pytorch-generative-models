import torch
from torch import nn


class EncoderConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.main_net = nn.Sequential(
            nn.Conv2d(params['channels'], self.params['n'], self.params['kernel_size']),
            nn.ELU(inplace=True),
            self._conv_block(1, 2),  # 32
            self._conv_block(2, 4),  # 16
            self._conv_block(4, 8),  # 8
            self._conv_block(8, 8),  # 4
            nn.Flatten()
        )
        self.var = nn.Linear(params['n'] * 128 + params['classes'], params['z'])
        self.mean = nn.Linear(params['n'] * 128 + params['classes'], params['z'])

    def _conv_block(self, n1, n2):
        return nn.Sequential(
            nn.Conv2d(
                self.params['n'] * n1,
                self.params['n'] * n2,
                self.params['kernel_size'],
                2,
                1
            ),
            nn.ELU(inplace=True),
        )

    def forward(self, x, y):
        main_out = self.main_net(x)
        main_out = torch.concat([main_out, y], dim=1)
        varlog = self.var(main_out)
        mean = self.mean(main_out)
        return varlog, mean


class DecoderConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.lin = nn.Sequential(
            nn.Linear(params['z'] + params['classes'], params['n']*128),
            nn.ELU(inplace=True),
        )
        self.net = nn.Sequential(
            self._conv_block(8, 8),
            self._conv_block(8, 4),
            self._conv_block(4, 2),
            self._conv_block(2, 1),
            nn.Conv2d(params['n'], params['channels'], 1, 1, 'same'),
            nn.Sigmoid(),
        )

    def _conv_block(self, n1, n2):
        return nn.Sequential(
            nn.Conv2d(
                self.params['n'] * n1,
                self.params['n'] * n2,
                self.params['kernel_size'],
                1,
                'same'
            ),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, z, y):
        x = torch.concat([z, y], 1)
        lin = self.lin(x)
        lin = torch.reshape(lin, [-1, self.params['n'] * 8, 4, 4])
        return self.net(lin)
