"""Sequence encoders and output heads for the v2 study.

Heads
  bce16  : 16 independent logits, one per future day (the original formulation).
  hazard : 1 logit for the current trend y[0] plus 15 discrete-time hazard logits
           h_k = P(first reversal at step k | none before), k = 1..15. This matches the
           label structure [a, a, ..., a, b, ..., b] exactly.
Both heads expose `trend_probs` (P(down) per future day) and `event_probs(m)`
(P(first reversal within steps 1..m)) so they can be evaluated identically.
"""
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, n_features, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x):
        _, h = self.gru(x)
        return self.drop(h[-1])


class TCNEncoder(nn.Module):
    """Causal dilated convolutions over time (kernel 3, dilations 1, 2, 4, 8, 16)."""

    def __init__(self, n_features, hidden=48, dropout=0.2, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        layers, ch = [], n_features
        for d in dilations:
            layers.append(_CausalBlock(ch, hidden, d, dropout))
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class _CausalBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(cin, cout, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + self.res(x))


class MLPEncoder(nn.Module):
    """Uses only the last few days (no sequence model) - a deliberately simple neural baseline."""

    def __init__(self, n_features, hidden=64, dropout=0.2, last=5):
        super().__init__()
        self.last = last
        self.net = nn.Sequential(nn.Linear(n_features * last, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x[:, -self.last:].flatten(1))


ENCODERS = {'GRU': GRUEncoder, 'TCN': TCNEncoder, 'MLP': MLPEncoder}


class ReversalNet(nn.Module):
    def __init__(self, encoder, n_features, horizon=16, head='hazard', **enc_kw):
        super().__init__()
        self.encoder = ENCODERS[encoder](n_features, **enc_kw)
        self.head_type = head
        self.horizon = horizon
        self.head = nn.Linear(self.encoder.out_dim, horizon)  # both heads use `horizon` outputs

    def forward(self, x):
        return self.head(self.encoder(x))

    # ---- losses -------------------------------------------------------
    def loss(self, logits, Y, first, rev):
        if self.head_type == 'bce16':
            return nn.functional.binary_cross_entropy_with_logits(logits, Y)
        trend_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], first)
        log_h = nn.functional.logsigmoid(logits[:, 1:])      # log h_k, k = 1..15
        log_1mh = nn.functional.logsigmoid(-logits[:, 1:])   # log (1 - h_k)
        steps = torch.arange(1, self.horizon, device=logits.device).unsqueeze(0)
        r = rev.unsqueeze(1)
        survived = (steps < r).float()                       # no reversal at these steps
        happened = (steps == r).float()                      # reversal exactly here
        nll = -(survived * log_1mh + happened * log_h).sum(1) / (self.horizon - 1)
        return trend_loss + nll.mean()

    # ---- probabilities ------------------------------------------------
    def trend_probs(self, logits):
        """P(down-trend) for each of the `horizon` future days."""
        if self.head_type == 'bce16':
            return torch.sigmoid(logits)
        a = torch.sigmoid(logits[:, :1])
        survive = torch.cumprod(torch.sigmoid(-logits[:, 1:]), dim=1)   # S_k = prod_{j<=k}(1-h_j)
        survive = torch.cat([torch.ones_like(a), survive], dim=1)
        return a * survive + (1 - a) * (1 - survive)

    def event_probs(self, logits, m):
        """P(first trend reversal within steps 1..m)."""
        if self.head_type == 'bce16':
            p = torch.sigmoid(logits)
            p0, pm = p[:, 0], p[:, m]
            return p0 * (1 - pm) + (1 - p0) * pm  # P(y_m != y_0), assuming independence
        return 1 - torch.prod(torch.sigmoid(-logits[:, 1:m + 1]), dim=1)
