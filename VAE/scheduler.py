import torch
from torch import nn
import torch.nn.functional as F
import VAE.models as models
import VAE.initializers as initializers
import utils


class MNISTScheduler(utils.Scheduler):
    def __init__(self, root_path, name, feeder, models, params, snapshot=utils.SaveMode.NO_VERSIONING):
        super().__init__(root_path, name, feeder, models, params, snapshot=snapshot)
        self.test_z = torch.randn((16, params['z']), device='cuda')
        self.normal = torch.randn((params['batch_size'], params['z']), device='cuda')
        self.loss = nn.BCELoss(reduction='sum')
        self.preprocess_x = lambda x: x
        if params['baseline']:
            self.preprocess_x = lambda x: x.view(-1, 784)
        self.eval_size = [16, -1, 28, 28] if params['baseline'] else [16, -1, 32, 32]

    def post_init(self):
        self.enc, self.dec = self.models

    def train_op(self, data):
        x, _ = data
        x = x.cuda()
        x = self.preprocess_x(x)

        self.normal.normal_()

        self.optims.zero_grad()

        varlog, mean = self.enc(x)
        var = torch.exp(varlog)
        z = mean + self.normal * var
        g = self.dec(z)
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
        out = self.dec(self.test_z)
        out = out * 2 - 1
        out = torch.reshape(out, self.eval_size)
        return out

    def make_optimizers(self):
        parameters = list(self.models[0].parameters()) + list(self.models[1].parameters())
        self.optims = torch.optim.Adam(parameters, self.params['lr'])
