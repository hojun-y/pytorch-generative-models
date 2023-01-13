import os
import torchvision.transforms as xforms
from torch.utils.data import DataLoader
import utils

CHUNK_SIZE = 10000

path = os.path.join('data', 'datasets')
source_path = os.path.join(path, 'celeba/Dataset/Train/')
dataset_path = os.path.join(path, 'cache')


class Data(utils.DataGenerator):
    def __init__(self, params):
        super().__init__(params)
        dataset = utils.CachedImageDataset(
            dataset_path,
            f'celeba-{params["res"]}px'
        )
        self.loader = DataLoader(dataset, batch_size=params['batch_size'])


def get_dataloader(params, res=64):
    params['classes'] = 2
    params['res'] = res
    params['channels'] = 3
    params['data_name'] = f"celeba64"
    return Data


def _square_crop(x):
    shape = x.shape
    center = (0.5 * shape[1], 0.5 * shape[2])
    size = min(center)
    x = x[
        :,
        int(center[0] - size):int(center[0] + size),
        int(center[1] - size):int(center[1] + size),
        ]
    return x


if __name__ == '__main__':
    writer = utils.CachedImageDatasetWriter(source_path, CHUNK_SIZE)
    writer.save(
        xforms.Compose([
            xforms.ToTensor(),
            xforms.Lambda(lambda x: _square_crop(x)),
            xforms.Resize([64, 64]),
        ]),
        (3, 64, 64),
        "celeba-64px",
        dataset_path,
        True
    )
