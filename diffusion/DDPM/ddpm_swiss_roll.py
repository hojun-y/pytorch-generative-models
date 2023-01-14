from diffusion.diffusion_tools import DiffusionTools
from diffusion.swiss_roll import SwissRollData
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * params['t_dim'], 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 2),
            Rearrange('b (c h w) -> b c h w', c=2, h=1, w=1)
        )

    def forward(self, x, t):
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = torch.concat([x, t], dim=1)
        return self.net(x)


params = {
    "t_dim": 10,
    "t_max": 100,
    "w_base": 1e-3,
    "s": 0.008,
    "x_shape": [2, 1, 1],
    "steps": 10000,
    "batch_size": 1024,
    "lr": 0.005,
}

data = SwissRollData(.8)
diff_tools = DiffusionTools(params)
model = Network().cuda()
optim = torch.optim.Adam(model.parameters(), lr=params['lr'])
batch_size = params['batch_size']
t_max = params['t_max']
criterion = nn.MSELoss()

cum_loss = 0.0
cum_steps = 0
for i in range(params['steps']):
    batch = rearrange(data.get_data(batch_size), 'b (c h w) -> b c h w', c=2, h=1, w=1)
    t = torch.randint(0, t_max, (batch_size,), device='cuda')
    t_embed = diff_tools.get_position_embedding(t)
    batch = diff_tools.sample_q(batch, t)

    optim.zero_grad()
    pred = model(batch, t_embed)
    loss = criterion(
        diff_tools.gaussian,
        pred
    )
    loss.backward()
    cum_loss += loss.item()
    cum_steps += 1
    optim.step()

    if i % 100 == 99:
        print(f'Steps #{i+1:05}\tLoss: {cum_loss / cum_steps}')

test_pts = 100
result = torch.randn([test_pts, 2, 1, 1], device='cuda')

result = diff_tools.sample_p(result, model, t_max)
result = rearrange(result, 'b c h w -> b (c h w)')
print(result.min(), result.max())

data.viz_data(result)
