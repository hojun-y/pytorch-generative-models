import torch
from torch import nn
import torch.nn.functional as F

import utils
from functools import partial
from datasets.flowers import get_dataloader
from diffusion.DDPM.scheduler import DDPMScheduler
from diffusion.DDPM.models import UNet
import matplotlib.pyplot as plt

params = {
    'batch_size': 16,
    'lr': 1e-5,
    'betas': (0.5, 0.999),
    'filters': (64, 64, 128, 256),
    'groups': 8,
    't_dim': 50,
    't_max': 300,
    's': 0.008,
    'x_shape': [1, 32, 32],
    'w_base': 1e-4,
    'activation': partial(nn.ELU, inplace=True),
    'steps': 10000,
    'log_every': 100,
    'eval_every': 1000,
    'save_every': 5e4,
}

dataloader = get_dataloader(params, 32, False)
utils.handle_cmd_params(params)

scheduler = \
    DDPMScheduler(f'data/', 'DDPM-' + params['data_name'],
                  dataloader,
                  UNet,
                  params, utils.SaveMode.STEPS)

if params['train']:
    scheduler.make_optimizers()
    scheduler.train()
else:
    with torch.no_grad():
        result = scheduler.eval_op()
        img = scheduler.recorder.create_img(result)
        plt.show()
