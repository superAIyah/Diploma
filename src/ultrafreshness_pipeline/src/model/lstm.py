import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    Implementation of LSTM-based time-series classifier.
    Took from here:
    https://www.kaggle.com/code/purplejester/a-simple-lstm-based-time-series-classifier
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        h0, c0 = self.init_hidden(x, device)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(device) for t in (h0, c0)]
