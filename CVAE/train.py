import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from CVAE.scheduler import CVAEScheduler
import CVAE.models as models
from datasets.celeba import get_dataloader

import utils
import initializers


params = {
    'batch_size': 32,
    'z': 32,
    'n': 64,
    'kernel_size': 3,
    'lr': 1e-4,
    'steps': 2e5,
    'log_every': 25,
    'eval_every': 2000,
    'eval_tile': 10,
    'save_every': 5e4,
}

dataloader = get_dataloader(params, 64)
utils.handle_cmd_params(params)


scheduler = \
    CVAEScheduler(f'data/', 'CVAE-' + params['data_name'],
                  dataloader,
                  (models.EncoderConv, models.DecoderConv),
                  params, utils.SaveMode.STEPS
                  )

if params['train']:
    scheduler.init((initializers.default_init, initializers.default_init))
    scheduler.make_optimizers()
    scheduler.train()
else:
    scheduler.load_weights(params['steps'])
    with torch.no_grad():
        RES = 10
        z = torch.randn([RES * RES, params['z']]).cuda()
        for i in range(params['classes']):
            ohe = F.one_hot(
                torch.full((RES * RES,), i, dtype=torch.long),
                params['classes']
            ).to(torch.float32).cuda()

            generator = scheduler.dec
            generator.eval()
            out = generator(z, ohe)
            img = utils.make_grid(out, RES, 0).cpu().numpy().transpose((1, 2, 0))
            fig = plt.figure(figsize=(RES, RES), dpi=params['res'])
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img)
            plt.savefig(f'data/out/CVAE/{params["data_name"]}-{i:03}.png')
