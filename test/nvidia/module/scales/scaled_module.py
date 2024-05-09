import torch
from torch import nn


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def scaled_rwkv_feedforward(self, hidden, state=None):
    if hidden.size(1) == 1 and state is not None:
        shifted = state[0][:, :, self.layer_id]
    else:
        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[0][:, :, self.layer_id]
    key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
    receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

    key = torch.square(torch.relu(self.key(key))) / (self.scales.view(1, 1, -1).to(key.device))
    value = self.value(key)
    receptance = torch.sigmoid(self.receptance(receptance))

    if state is not None:
        state[0][:, :, self.layer_id] = hidden[:, -1]

    return receptance * value, state
