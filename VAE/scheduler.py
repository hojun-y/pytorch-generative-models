import torch
from torch import nn
import torch.nn.functional as F
import VAE.models as models
import VAE.initializers as initializers
import utils


class MNISTScheduler(utils.Scheduler):
    def __init__(self, root_path, name, feeder, models, params, snapshot=utils.SaveMode.NO_VERSIONING):
        super().__init__(root_path, name, feeder, models, params, snapshot=snapshot)
        self.inv_bs_half = .5 / params['batch_size']
        self.test_z = \
            torch.randn((16, params['z']), device='cuda') if params['baseline'] else \
            torch.randn((16, params['z'], 4, 4), device='cuda')
        self.normal = \
            torch.randn((params['batch_size'], params['z']), device='cuda') if params['baseline'] else \
            torch.randn((params['batch_size'], params['z'], 4, 4), device='cuda')
        self.enc, self.dec = models
        self.loss = nn.BCELoss(reduction='sum')
        self.preprocess_x = lambda x: x
        if params['baseline']:
            self.preprocess_x = lambda x: x.view(-1, 784)
        self.eval_size = [16, -1, 28, 28] if params['baseline'] else [16, -1, 32, 32]

    def train_op(self, data):
        enc, dec = self.models
        x, _ = data
        x = x.cuda()
        x = self.preprocess_x(x)

        self.normal.normal_()

        self.optims.zero_grad()

        varlog, mean = enc(x)
        var = torch.exp(varlog)
        z = mean + self.normal * var
        g = dec(z)
        g = torch.clamp(g, 0., 1.)
        bce = self.loss(g, x)
        kld = 0.5 * torch.sum(var + mean.square() - varlog - 1)
        loss = bce + kld
        loss.backward()

        self.optims.step()

        return {
            "BCE": bce,
            "KLD": kld,
            "Loss": loss
        }

    def eval_op(self):
        _, dec = self.models
        out = dec(self.test_z)
        out = out * 2 - 1
        out = torch.reshape(out, self.eval_size)
        return out

    def make_optimizers(self):
        parameters = list(self.models[0].parameters()) + list(self.models[1].parameters())
        self.optims = torch.optim.Adam(parameters, self.params['lr'])
