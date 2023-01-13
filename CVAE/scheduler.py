import torch
from torch import nn
import torch.nn.functional as F
import utils
import numpy as np


class CVAEScheduler(utils.Scheduler):
    def __init__(self, root_path, name, feeder, models, params, snapshot=utils.SaveMode.NO_VERSIONING):
        super().__init__(root_path, name, feeder, models, params, snapshot=snapshot)
        self.test_z = torch.randn((params['classes'] * params['eval_tile'], params['z']), device='cuda')
        self.test_y = F.one_hot(torch.LongTensor(
            np.tile(np.arange(params['classes']), (params['eval_tile'],))
        )).to(torch.float32).cuda()
        self.normal = torch.randn((params['batch_size'], params['z']), device='cuda')
        self.loss = nn.BCELoss()
        self.eval_size = [params['classes'] * params['eval_tile'], -1, params['res'], params['res']]

    def post_init(self):
        self.enc, self.dec = self.models

    def train_op(self, data):
        x, y = data
        x, y = x.cuda(), y.cuda()
        y = F.one_hot(y, self.params['classes']).to(torch.float32)

        self.normal.normal_()
        self.optims.zero_grad()
        varlog, mean = self.enc(x, y)
        var = torch.exp(varlog)
        z = mean + self.normal * var
        g = self.dec(z, y)
        g = torch.clamp(g, 1e-8, 1. - 1e-8)
        bce = self.loss(g, x)
        kld = 0.5 * torch.mean(torch.sum(var + mean.square() - varlog - 1, dim=1))
        kld = torch.clamp(kld, -1e2, 1e2)
        loss = bce + kld
        loss.backward()

        self.optims.step()

        return {
            "BCE": bce,
            "KLD": kld,
            "Loss": loss
        }

    def eval_op(self):
        out = self.dec(self.test_z, self.test_y)
        out = out * 2 - 1
        out = torch.reshape(out, self.eval_size)
        return out

    def make_optimizers(self):
        parameters = list(self.models[0].parameters()) + list(self.models[1].parameters())
        self.optims = torch.optim.Adam(parameters, self.params['lr'])
