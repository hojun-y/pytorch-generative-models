import os
import torchvision.transforms as xforms
from torch.utils.data import DataLoader
import utils

CHUNK_SIZE = 5000

path = os.path.join('data', 'datasets')
source_path = os.path.join(path, 'flowers')
dataset_path = os.path.join(path, 'cache')


class Data(utils.DataGenerator):
    def __init__(self, params):
        super().__init__(params)
        dataset = utils.CachedImageDataset(
            dataset_path,
            f'flowers-{params["res"]}px'
        )
        self.loader = DataLoader(dataset, batch_size=params['batch_size'])


def get_dataloader(params, res=64):
    params['classes'] = 16
    params['res'] = res
    params['channels'] = 3
    params['data_name'] = f"flower{res}"
    return Data


def _grey2col(x):
    if x.shape[0] == 1:
        return x[0].repeat(3, 1, 1)
    else:
        return x


if __name__ == '__main__':
    writer = utils.CachedImageDatasetWriter(source_path, CHUNK_SIZE)
    writer.save(
        xforms.Compose([
            xforms.ToTensor(),
            xforms.Resize([64, 64]),
            xforms.Lambda(lambda x: _grey2col(x)),
        ]),
        (3, 64, 64),
        "flowers-64px",
        dataset_path,
        True
    )

    writer = utils.CachedImageDatasetWriter(source_path, CHUNK_SIZE)
    writer.save(
        xforms.Compose([
            xforms.ToTensor(),
            xforms.Resize([32, 32]),
            xforms.Lambda(lambda x: _grey2col(x)),
        ]),
        (3, 32, 32),
        "flowers-32px",
        dataset_path,
        True
    )
