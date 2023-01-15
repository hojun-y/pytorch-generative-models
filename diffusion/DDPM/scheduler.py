import torch
from torch import nn
import torch.nn.functional as F

import diffusion.diffusion_tools as difftools
import utils


class DDPMScheduler(utils.Scheduler):

    def __init__(self, root_path, name, feeder, models, params, snapshot=utils.SaveMode.NO_VERSIONING):
        super().__init__(root_path, name, feeder, models, params, snapshot=snapshot)
        self.diffusion_tools = difftools.DiffusionTools(params)
        self.test_noise = torch.randn((8, params['channels'], 32, 32), device='cuda')
        self.test_t = self.diffusion_tools.get_position_embedding(
            torch.full((8,), params['t_max'], dtype=torch.long, device='cuda'))

        self.criterion = nn.MSELoss()

        self.batch_size = params['batch_size']
        self.t_max = params['t_max'] + 1

    def train_op(self, data):
        x, _ = data
        x = x.cuda()
        t = self.diffusion_tools.get_position_embedding(
            torch.randint(0, self.t_max, (self.batch_size,), dtype=torch.long, device='cuda')
        )

        self.optims.zero_grad()
        pred = self.models(x, t)
        loss = self.criterion(pred, self.diffusion_tools.gaussian)
        loss.backward()
        self.optims.step()

        return {
            "loss": loss
        }

    def eval_op(self):
        t_half = self.params['t_max'] // 2
        result_half = self.diffusion_tools.sample_p(self.test_noise, self.models, self.params['t_max'], t_half)
        result = self.diffusion_tools.sample_p(result_half, self.models, t_half)
        result = torch.concat([result_half, result], dim=0)
        return result

    def make_optimizers(self):
        self.optims = torch.optim.Adam(self.models.parameters(), self.params['lr'], self.params['betas'])
