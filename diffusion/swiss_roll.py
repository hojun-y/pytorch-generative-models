import matplotlib.pyplot as plt
import torch
import numpy as np


class SwissRollData:
    def __init__(self, m=1.):
        self.twopi = torch.pi * 2
        self.scale = m / self.twopi
        self.m = m

        viz_pts = np.linspace(0, self.twopi, 100)
        self.viz_x = self.scale * viz_pts * np.sin(viz_pts)
        self.viz_y = self.scale * viz_pts * np.cos(viz_pts)

    def get_data(self, batch_size, device='cuda'):
        pos = torch.rand(batch_size, device=device, requires_grad=False) * self.twopi
        batch = torch.stack([torch.sin(pos), torch.cos(pos)]) * pos
        batch *= self.scale
        batch = torch.transpose(batch, 1, 0)
        return batch

    def viz_data(self, x, show=True):
        with torch.no_grad():
            x = x.cpu().numpy().T
            plt.plot(self.viz_x, self.viz_y, 'k')
            plt.xlim([-self.m - .1, self.m + .1])
            plt.ylim([-self.m - .1, self.m + .1])
            plt.scatter(x[0], x[1], c='r', marker='.')
            if show:
                plt.show()
