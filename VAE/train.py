from VAE.scheduler import MNISTScheduler
import utils
import VAE.models as models
import VAE.initializers as initializers
import torchvision.transforms as xforms
import torchvision.datasets
from torch.utils.data import DataLoader
import os


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
    'batch_size': 64,
    'z': 64,
    'n': 256,
    'lr': 0.0008,
    'steps': 8e4,
    'log_every': 20,
    'eval_every': 400,
    'save_every': 5e3,
    'baseline': True,
}


conv_params = {
    'batch_size': 64,
    'z': 8,
    'n': 32,
    'kernel_size': 3,
    'lr': 0.001,
    'steps': 8e4,
    'log_every': 20,
    'eval_every': 250,
    'save_every': 5e3,
    'baseline': False,
}


CONV = True

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

scheduler.init((initializers.init_model, initializers.init_model))
scheduler.make_optimizers()
scheduler.train()
