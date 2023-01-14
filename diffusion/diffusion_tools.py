import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class DiffusionTools:
    def __init__(self, params):
        # parameters for sinusoidal position embedding
        self.t_dim = params['t_dim'] * 2
        self.t_max = params['t_max']
        self.w_base = params['w_base']

        freq_evens = np.arange(params['t_dim']) * 4 / self.t_dim
        freq_odds = freq_evens - (1 / self.t_dim)
        self.freq_evens = torch.tensor(np.power(self.w_base, freq_evens), device='cuda')
        self.freq_odds = torch.tensor(np.power(self.w_base, freq_odds), device='cuda')

        # parameters for noise generators
        cos_arg_scale = (0.5 * torch.pi / (1 + params['s']))
        self.alpha_cum_lut = torch.linspace(0., 1., self.t_max + 1) + params['s']
        self.alpha_cum_lut *= cos_arg_scale
        self.alpha_cum_lut.cos_().square_()
        self.alpha_cum_lut = self.alpha_cum_lut / self.alpha_cum_lut[0]  # t = 0 ... T
        self.q_mean_lut = torch.sqrt(self.alpha_cum_lut).cuda()  # t = 0 ... T
        self.q_var_lut = torch.sqrt(1. - self.alpha_cum_lut).cuda()  # t = 0 ... T
        self.beta_lut = 1. - (self.alpha_cum_lut[1:] / self.alpha_cum_lut[:-1])
        self.beta_lut.clamp_(0., .999)

    def get_position_embedding(self, t):
        return rearrange(torch.stack((
            ((t.outer(self.freq_odds)).cos()),
            (t.outer(self.freq_evens)).sin()
        )), 'x b y -> b (y x)')

    def add_noise(self, x, t):
        noise = torch.randn_like(x, device='cuda')
        noise *= self.q_var_lut.index_select(0, t).view(-1, 1, 1, 1)
        return x * self.q_mean_lut.index_select(0, t).view(-1, 1, 1, 1) + noise
