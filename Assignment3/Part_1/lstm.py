import torch
import torch.nn as nn


################################################################################


class LSTM(nn.Module):

    def __init__(
        self,
        seq_length: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
    ):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.gx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.gh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ix = nn.Linear(input_dim, hidden_dim, bias=True)
        self.ih = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.fx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ox = nn.Linear(input_dim, hidden_dim, bias=True)
        self.oh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ph = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_length = x.size(0), x.size(1)

        h_last = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_last = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        for t in range(input_length):
            x_cur = x[:, t, :]
            g_t = torch.tanh(self.gx(x_cur) + self.gh(h_last))
            i_t = torch.sigmoid(self.ix(x_cur) + self.ih(h_last))
            f_t = torch.sigmoid(self.fx(x_cur) + self.fh(h_last))
            o_t = torch.sigmoid(self.ox(x_cur) + self.oh(h_last))
            c_last = g_t * i_t + c_last * f_t
            h_last = torch.tanh(c_last) * o_t

        p = self.ph(h_last)
        y = torch.softmax(p, dim=-1)

        return y
