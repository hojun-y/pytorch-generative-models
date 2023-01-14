import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange, Reduce


class DiffusionTools:
    def __init__(self, params):
        # parameters for sinusoidal position embedding
        self.t_dim = params['t_dim'] * 2
        self.t_max = params['t_max']
        self.w_base = params['w_base']

        freq_evens = np.arange(params['t_dim']) * 4 / self.t_dim
        freq_odds = freq_evens - (1 / self.t_dim)
        self.freq_evens = torch.tensor(np.power(self.w_base, freq_evens), dtype=torch.float32, device='cuda')
        self.freq_odds = torch.tensor(np.power(self.w_base, freq_odds), dtype=torch.float32, device='cuda')

        # parameters for noise generators
        cos_arg_scale = (0.5 * torch.pi / (1 + params['s']))
        self.gaussian = torch.randn([params['batch_size']] + params['x_shape']).cuda()
        self.alpha_cum_lut = torch.linspace(0., 1., self.t_max + 1) + params['s']
        self.alpha_cum_lut *= cos_arg_scale
        self.alpha_cum_lut.cos_().square_()
        self.alpha_cum_lut = self.alpha_cum_lut / self.alpha_cum_lut[0]  # t = 0 ... T
        self.q_mean_lut = torch.sqrt(self.alpha_cum_lut)  # t = 0 ... T
        self.q_var_lut = torch.sqrt(1. - self.alpha_cum_lut)  # t = 0 ... T

        beta_lut = 1. - (self.alpha_cum_lut[1:] / self.alpha_cum_lut[:-1])  # t = 1 ... T
        beta_lut.clamp_(0., .999)
        self.epsilon_scale_lut = beta_lut / self.q_var_lut[1:]
        self.sqrt_beta_lut = beta_lut.sqrt_()
        self.inv_sqrt_alpha_lut = 1. / (1 - beta_lut).sqrt_()

        self.q_mean_lut = self.q_mean_lut.cuda()
        self.q_var_lut = self.q_var_lut.cuda()
        self.sqrt_beta_lut = self.sqrt_beta_lut.cuda()
        self.inv_sqrt_alpha_lut = self.inv_sqrt_alpha_lut.cuda()
        self.epsilon_scale_lut = self.epsilon_scale_lut.cuda()

    def get_position_embedding(self, t):
        return rearrange(torch.stack((
            ((t.outer(self.freq_odds)).cos()),
            (t.outer(self.freq_evens)).sin()
        )), 'x b y -> b (y x)')

    @staticmethod
    def mul_(x, s):
        return einsum(x, s, 'b c h w, b -> b c h w')

    def sample_q(self, x, t):
        self.gaussian.normal_()
        # diffused = self.mul_(x, self.q_mean_lut.index_select(0, t)) + \
        #     self.mul_(self.gaussian, self.q_var_lut.index_select(0, t))
        diffused = self.mul_(x, self.q_mean_lut.index_select(0, t))
        return diffused

    def sample_p(self, x, p, t_from, t_to=0):
        with torch.no_grad():
            noise = torch.empty_like(x, device='cuda')
            for t in torch.arange(t_from - 1, t_to, -1, device='cuda'):
                t_idx = t - 1
                noise.normal_()
                embed = self.get_position_embedding(t.view(-1))
                infer = p(x, torch.tile(embed, [x.shape[0], 1]))
                mean = self.mul_(
                    (x - self.mul_(infer, self.epsilon_scale_lut.index_select(0, t_idx))),
                    self.inv_sqrt_alpha_lut.index_select(0, t_idx),
                )

                x = mean + self.mul_(noise, self.sqrt_beta_lut.index_select(0, t_idx))
        return x
