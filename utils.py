from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
import torch
from collections.abc import Iterable
import matplotlib.pyplot as plt
import matplotlib as m
import datetime
from enum import Enum
import os
from PIL import Image
import argparse
import pickle
import random


parser_ = argparse.ArgumentParser()
parser_.add_argument('--train', action='store_true')
parser_.add_argument('--eval', type=int, default=0)


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
        self.models = \
            [model_base(params).cuda() for model_base in models] if isinstance(models, Iterable) \
                else models(params).cuda()
        self.optims = None
        self.params = params
        self.version_ctrl_mode = snapshot
        self.post_init()

    def post_init(self):
        pass

    def init(self, initializers):
        if isinstance(self.models, Iterable):
            for m, i in zip(self.models, initializers):
                m.apply(i)
        else:
            self.models.apply(initializers)

    def load_weights(self, model_id=None):
        if isinstance(self.models, Iterable):
            for m in self.models:
                self.recorder.load_weights(m, m.__class__.__name__, model_id)
        else:
            self.recorder.load_weights(self.models, self.models.__class__.__name__, model_id)

    def train(self):
        self.recorder.init_tensorboard()

        if isinstance(self.models, Iterable):
            for m in self.models:
                m.train()
        else:
            self.models.train()

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

        if isinstance(self.models, Iterable):
            for m in self.models:
                self.recorder.save_weights(m.state_dict(), m.__class__.__name__, self.version_ctrl_mode)
        else:
            self.recorder.save_weights(self.models.state_dict(), self.models.__class__.__name__, self.version_ctrl_mode)

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


class ImageChunk:
    def __init__(self, size):
        self.size = size
        self.data = torch.empty(size)
        self.labels = torch.empty(self.size[0], dtype=torch.long)

    def add(self, data, label, cur):
        self.data[cur] = data
        self.labels[cur] = label

    def __iter__(self):
        for i in range(self.size[0]):
            yield self.data[i], self.labels[i]

    def __len__(self):
        return self.size[0]


class CachedImageDatasetWriter:
    def __init__(self, root_path, chunk_size):
        self.root_path = root_path
        self.n = 0
        self.n_chunks = 0
        self.chunk_idx = 0
        self.chunk_size = chunk_size
        self.chunk_sizes = []
        self.labels_lookup = {}
        self.info = {}
        self.label_id = 0
        self.global_cursor = 0

    def save(self, xform, shape, name, out_path="", shuffle=True):
        files = []

        for path in os.listdir(self.root_path):
            interm_path = os.path.join(self.root_path, path)
            label = path
            if os.path.isdir(interm_path):
                self.labels_lookup[self.label_id] = label
                for p, _, f in os.walk(interm_path):
                    for file in f:
                        full_path = os.path.join(p, file)
                        files.append((full_path, self.label_id))
                self.label_id += 1
        self.n = len(files)
        self.chunk_sizes = [self.chunk_size] * (self.n // self.chunk_size)
        if self.n % self.chunk_size:
            self.chunk_sizes += [self.n % self.chunk_size]
        self.n_chunks = len(self.chunk_sizes)
        self.info = {"n": self.n, "sizes": self.chunk_sizes, "shape": shape}

        print(f"[INFO] {self.n} files labeled with {self.label_id} classes found.")
        print(f"[INFO] {self.n_chunks} chunks will be generated.")

        cursor = 0
        chunk = ImageChunk((self.chunk_sizes[self.chunk_idx],) + shape)

        if shuffle:
            random.shuffle(files)

        for i, (full_path, label) in enumerate(files):
            img = Image.open(full_path)
            img = xform(img)
            chunk.add(img, label, cursor)
            self.global_cursor += 1
            cursor += 1
            if cursor == self.chunk_sizes[self.chunk_idx]:
                cursor = 0
                torch.save(
                    chunk.data,
                    os.path.join(out_path, f"{name}.{self.chunk_idx:03}.x.pt")
                )
                torch.save(
                    chunk.labels,
                    os.path.join(out_path, f"{name}.{self.chunk_idx:03}.y.pt")
                )
                self.chunk_idx += 1
                if self.chunk_idx < self.n_chunks:
                    chunk = ImageChunk((self.chunk_sizes[self.chunk_idx],) + shape)

            if i % 25 == 0:
                print(f"\r[INFO] {i / self.n * 100.:3.0f}% done --- {i:08} files processed.", end="")

        with open(os.path.join(out_path, f"{name}.info.dat"), 'wb') as f:
            pickle.dump(self.info, f)

        with open(os.path.join(out_path, f"{name}.lut.dat"), 'wb') as f:
            pickle.dump(self.labels_lookup, f)

        print("\n[INFO] Done.")


class CachedImageDataset(IterableDataset):
    def __init__(self, path, name):
        super().__init__()
        self.name = name
        self.path = path
        self.data = []
        self.labels = []
        self.chunk_sizes = {}
        self.data_shape = ()
        self.n_files = 0
        self.n = 0
        self.cur_id = 0
        self.data_selector = 0

        self.lut = {}

        with open(os.path.join(path, f"{name}.info.dat"), 'rb') as f:
            info = pickle.load(f)

        with open(os.path.join(path, f"{name}.lut.dat"), 'rb') as f:
            self.lut = pickle.load(f)
        self.chunk_sizes = info['sizes']
        self.n = info['n']
        self.n_files = len(self.chunk_sizes)
        self.data_shape = (self.chunk_sizes[0],) + info['shape']
        self.data = [
            torch.zeros(self.data_shape).share_memory_(),
            torch.zeros(self.data_shape).share_memory_()
        ]
        self.labels = [
            torch.zeros([self.chunk_sizes[0]], dtype=torch.long).share_memory_(),
            torch.zeros([self.chunk_sizes[0]], dtype=torch.long).share_memory_()
        ]

    def _next_file(self):
        while True:
            self.cur_id += 1
            self.cur_id %= self.n_files
            yield self.cur_id

    def _flip_selector(self):
        self.data_selector = 1 if not self.data_selector else 0

    def _get_inverted_selector(self):
        return 1 if not self.data_selector else 0

    def _load(self, init=False):
        chunk_id = 0 if init else self.cur_id
        sel = 0 if init else self._get_inverted_selector()
        filename = f"{self.name}.{chunk_id:03}"
        filename = os.path.join(self.path, filename)
        x = torch.load(filename + '.x.pt')
        y = torch.load(filename + '.y.pt')
        self.data[sel][0:self.chunk_sizes[chunk_id]] = x
        self.labels[sel][0:self.chunk_sizes[chunk_id]] = y

    def __len__(self):
        return self.n

    def __iter__(self):
        self._load(True)

        if self.n_files == 1:
            while True:
                for i in range(self.chunk_sizes[self.cur_id]):
                    yield self.data[self.data_selector][i], self.labels[self.data_selector][i]

        for _ in self._next_file():
            proc = mp.Process(target=self._load, args=(False,))
            proc.start()
            for i in range(self.chunk_sizes[self.cur_id]):
                yield self.data[self.data_selector][i], self.labels[self.data_selector][i]

            self._flip_selector()
            proc.join()

    def map_labels(self, x):
        x = x.cpu().numpy()
        return [self.lut[i] for i in x]


def handle_cmd_params(params):
    args = parser_.parse_args()
    if args.train:
        params['train'] = True
    elif args.eval:
        params['train'] = False
        params['steps'] = args.eval


def send_to_pyplot(x, imshow=True, normalized=True):
    if normalized:
        x = 0.5 * (x + 1)

    dpi = m.rcParams['figure.dpi']
    h, w, c = x.shape
    h, w = float(h) / dpi, float(w) / dpi
    fig = plt.figure(figsize=(w, h))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(x)
    if imshow:
        plt.show()
