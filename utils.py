from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import datetime
from enum import Enum


class SaveMode(Enum):
    NO_VERSIONING = 0
    STEPS = 1


class Recorder:
    def __init__(self, data_root, sess_name, flush_every=60):
        self.steps = 0
        self.name = sess_name
        self.data_root = data_root
        self.sess_name = sess_name
        self.flush_every = flush_every
        self.writer = None

    def init_tensorboard(self):
        self.writer = SummaryWriter(
            f'{self.data_root}/tensorboard/{self.sess_name}-{self._get_ts()}', flush_secs=self.flush_every)

    @staticmethod
    def _get_ts():
        return f'{int(datetime.datetime.now().timestamp()):012}'

    @staticmethod
    def create_img(data):
        shape = data.shape
        w = shape[0] // 2
        grid = make_grid(data, w, normalize=True, value_range=(-1, 1), padding=0)

        grid_shape = grid.shape
        fig = plt.figure(figsize=(grid_shape[2], grid_shape[1]), dpi=1)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = grid.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        return fig

    def advance(self):
        self.steps += 1

    def add_vals(self, data):
        print(f'\r[Step {self.steps:08}]', end='')
        for key, val in data.items():
            val = val.item()
            self.writer.add_scalar(key, val, self.steps)
            print(f'        {key}={val:<14.7f}', end='')

    def add_imgs(self, data):
        img = self.create_img(data)
        self.writer.add_figure('Results', img, global_step=self.steps)
        self.writer.flush()

    def close(self):
        self.writer.close()

    def save_weights(self, weights, name, snapshot=SaveMode.NO_VERSIONING):
        if snapshot == SaveMode.NO_VERSIONING:
            path = f'{self.data_root}/weights/{name}'
        elif snapshot == SaveMode.STEPS:
            path = f'{self.data_root}/weights/{name}-{self.steps:08}'
        else:
            raise AttributeError('Unknown SaveMode')

        torch.save(weights, path)

    def load_weights(self, model, name, model_id=None):
        path = f'{self.data_root}/weights/{name}' if model_id is None \
            else f'{self.data_root}/weights/{name}-{model_id:08}'
        model.load_state_dict(torch.load(path))


class Scheduler:
    def __init__(self, root_path, name, feeder, models, params, snapshot=SaveMode.NO_VERSIONING):
        self.data_feeder = feeder(params)
        self.recorder = Recorder(root_path, name)
        self.models = [model_base(params).cuda() for model_base in models]
        self.optims = None
        self.params = params
        self.version_ctrl_mode = snapshot
        self.post_init()

    def post_init(self):
        pass

    def init(self, initializers):
        for m, i in zip(self.models, initializers):
            m.apply(i)

    def load_weights(self, model_id=None):
        for m in self.models:
            self.recorder.load_weights(m, m.__class__.__name__, model_id)

    def train(self):
        self.recorder.init_tensorboard()

        for m in self.models:
            m.train()

        for data in self.data_feeder.data_feeder(self.params['steps']):
            out = self.train_op(data)
            self.recorder.advance()

            if self.recorder.steps % self.params['log_every'] == 0:
                self.recorder.add_vals(out)
            if self.recorder.steps % self.params['eval_every'] == 0:
                out = self.eval_op()
                self.recorder.add_imgs(out)
            if self.recorder.steps % self.params['save_every'] == 0:
                for m in self.models:
                    self.recorder.save_weights(m.state_dict(), m.__class__.__name__, self.version_ctrl_mode)

        for m in self.models:
            self.recorder.save_weights(m.state_dict(), m.__class__.__name__, self.version_ctrl_mode)

    def train_op(self, data):
        raise NotImplementedError
        return {}

    def eval_op(self):
        raise NotImplementedError
        return 0

    def make_optimizers(self):
        raise NotImplementedError


class DataGenerator:
    def __init__(self, params):
        self.params = params
        self.loader = None
        self.steps = 0
        self.batch_size = params['batch_size']

    def data_feeder(self, steps):
        while True:
            for d in self.loader:
                if self.steps == steps:
                    return d

                steps += 1
                yield d
