import torch
from torch import nn
import torch.nn.functional as F


def init_model(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0., 0.01)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0., 0.01)
