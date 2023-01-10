import numpy as np
import matplotlib.pyplot as plt
from VAE.scheduler import MNISTScheduler
import utils
import VAE.models as models
import VAE.initializers as initializers
import torchvision.transforms as xforms
import torchvision.datasets
from torch.utils.data import DataLoader
import os
import torch


class MNISTFeeder(utils.DataGenerator):
    def __init__(self, params):
        super().__init__(params)
        self.xform = xforms.Compose([
            xforms.ToTensor(),
        ])

        dataset = torchvision.datasets.MNIST('~/data/datasets/', transform=self.xform, download=True)
        self.loader = DataLoader(dataset, batch_size=params['batch_size'], drop_last=True)


class MNIST32Feeder(utils.DataGenerator):
    def __init__(self, params):
        super().__init__(params)
        self.xform = xforms.Compose([
            xforms.ToTensor(),
            xforms.Pad(2),
        ])

        dataset = torchvision.datasets.MNIST('~/data/datasets/', transform=self.xform, download=True)
        self.loader = DataLoader(dataset, batch_size=params['batch_size'], drop_last=True)


linear_params = {
    'batch_size': 128,
    'z': 3,
    'n': 256,
    'lr': 0.0008,
    'steps': 8e4,
    'log_every': 100,
    'eval_every': 1000,
    'save_every': 2e4,
    'baseline': True,
}


conv_params = {
    'batch_size': 64,
    'z': 3,
    'z_size': 4,
    'n': 64,
    'kernel_size': 3,
    'lr': 0.001,
    'steps': 1e5,
    'log_every': 250,
    'eval_every': 2e3,
    'save_every': 1e4,
    'baseline': False,
}


CONV = True
train = False

scheduler = \
    MNISTScheduler(f'{os.getenv("HOME")}/data/', 'VAE_Conv',
                   MNIST32Feeder,
                   (models.EncoderConv, models.DecoderConv),
                   conv_params, utils.SaveMode.STEPS
                   ) if CONV else \
    MNISTScheduler(f'{os.getenv("HOME")}/data/', 'VAE_Linear',
                   MNISTFeeder,
                   (models.EncoderLinear, models.DecoderLinear),
                   linear_params, utils.SaveMode.STEPS
                   )

if train:
    scheduler.init((initializers.init_model, initializers.init_model))
    scheduler.make_optimizers()
    scheduler.train()
else:
    RES = 32
    with torch.no_grad():
        scheduler.load_weights(300000)
        lin = np.linspace(-5, 5, RES, dtype=np.float32)
        for i, z0 in enumerate(lin):
            z = torch.full([RES * RES, 1], z0)
            xs, ys = np.meshgrid(lin, lin)
            interpolation = np.stack([xs.flatten(), ys.flatten()], 1)
            z_interpolation = torch.concat([z, torch.tensor(interpolation)], 1).cuda()
            generator = scheduler.dec
            generator.eval()
            out = generator(z_interpolation)
            img = utils.make_grid(out, RES, 0).cpu().numpy().transpose((1, 2, 0))
            fig = plt.figure(figsize=(RES, RES), dpi=32)
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img)
            plt.savefig(f'/out/VAE/{i:03}.png', facecolor='black')
