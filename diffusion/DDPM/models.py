import torch
from torch import nn
import torch.nn.functional as F
from einops import einsum, rearrange


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, groups, time_dim, activation):
        super().__init__()
        self.activation = activation()

        self.residual_connection = \
            nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

        self.time_embedding_injector = nn.ModuleList([
            nn.Linear(time_dim, out_c, bias=False),
            nn.Linear(time_dim, out_c, bias=False)
        ])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.GroupNorm(groups, out_c),
            activation(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.GroupNorm(groups, out_c),
        )

    def forward(self, x, t):
        scale = self.time_embedding_injector[0](t)
        bias = self.time_embedding_injector[1](t)
        bias = rearrange(bias, 'b c -> b c 1 1')

        f = self.conv1(x)
        f = einsum(f, scale, 'b c h w, b c -> b c h w')
        f = f + bias
        f = self.conv2(f)

        f2 = self.residual_connection(x)
        f = f + f2
        f = self.activation(f)
        return f


class DownscaleBlock(nn.Module):
    def __init__(self, in_c, out_c, groups, time_dim, activation):
        super().__init__()
        self.net = ResidualBlock(in_c, out_c, groups, time_dim, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        return self.pool(self.net(x, t))


class UpscaleBlock(nn.Module):
    def __init__(self, in_c, out_c, groups, time_dim, activation):
        super().__init__()
        self.net = ResidualBlock(2 * in_c, out_c, groups, time_dim, activation)

    def forward(self, x, x_conn, t):
        x = torch.concat([x, x_conn], dim=1)
        x = self.net(x, t)
        return F.interpolate(x, scale_factor=2)


class UNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.filters = params['filters']
        self.groups = params['groups']
        self.time_dim = params['t_dim'] * 2
        self.activation = params['activation']

        self.input_transform = nn.ModuleList([
            ResidualBlock(params['channels'], self.filters[0], self.groups, self.time_dim, self.activation),
            ResidualBlock(self.filters[0], self.filters[0], self.groups, self.time_dim, self.activation),
        ])
        self.output_transform = nn.ModuleList([
            ResidualBlock(self.filters[0], self.filters[0], self.groups, self.time_dim, self.activation),
            nn.Conv2d(self.filters[0], params['channels'], 3, 1, padding='same'),
            self.activation()
        ])

        self.downsampling_blocks = nn.ModuleList(DownscaleBlock(
            self.filters[i], self.filters[i + 1], self.groups, self.time_dim, self.activation)
                                               for i in range(0, 3, 1))

        self.upsampling_blocks = nn.ModuleList(UpscaleBlock(
            self.filters[i + 1], self.filters[i], self.groups, self.time_dim, self.activation)
                                               for i in range(2, -1, -1))

        self.bottleneck_block = ResidualBlock(
            self.filters[-1], self.filters[-1], self.groups, self.time_dim, self.activation
        )

    def forward(self, x, t):
        x = self.input_transform[1](
            self.input_transform[0](x, t),
            t
        )

        d1 = self.downsampling_blocks[0](x, t)  # 16*16
        d2 = self.downsampling_blocks[1](d1, t)  # 8*8
        d3 = self.downsampling_blocks[2](d2, t)  # 4*4

        b = self.bottleneck_block(d3, t)  # 4*4
        u3 = self.upsampling_blocks[0](b, d3, t)  # 8*8
        u2 = self.upsampling_blocks[1](u3, d2, t)  # 16*16
        u1 = self.upsampling_blocks[2](u2, d1, t)  # 32*32

        y = self.output_transform[1](
            self.output_transform[0](u1, t),
        )

        return y

