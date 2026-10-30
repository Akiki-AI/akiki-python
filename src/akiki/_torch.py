"""CPU PyTorch graph for the original Akiki CNN/GRU release.

The frontend/CNN follow the Silero v5 design; see models/SILERO-MIT.txt.
Only imported when PyTorch inference is selected.
"""

import torch
from torch import nn
from torch.nn import functional as F


class AkikiModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # The exact deterministic Fourier/Hann buffer comes from the checkpoint.
        self.register_buffer("basis", torch.zeros(258, 1, 256))
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(129, 128, 3, padding=1),
                nn.Conv1d(128, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 128, 3, padding=1),
            ]
        )
        self.rnn = nn.GRU(128, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, audio, state):
        padded = F.pad(audio, (0, 64), mode="reflect")
        x = F.linear(padded.unfold(-1, 256, 128), self.basis[:, 0]).transpose(1, 2)
        x = (x[:, :129].square() + x[:, 129:].square()).sqrt()
        for layer in self.encoder:
            x = F.pad(x, (layer.padding[0], layer.padding[0]))
            windows = x.unfold(-1, layer.kernel_size[0], layer.stride[0])
            windows = windows.permute(0, 2, 1, 3).flatten(2)
            x = F.relu(F.linear(windows, layer.weight.flatten(1), layer.bias).transpose(1, 2))
        x, hidden = self.rnn(x.squeeze(-1).unsqueeze(1), state)
        return self.head(F.relu(x[:, 0])).squeeze(-1).sigmoid(), hidden
