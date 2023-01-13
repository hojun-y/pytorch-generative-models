from torch import nn


def default_init(m):
    pass


def init_model(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0., 0.01)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0., 0.01)
