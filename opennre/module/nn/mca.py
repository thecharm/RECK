from .net_utils import FC, MLP, Layernorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Attention ----
# ------------------------------

class ATT(nn.Module):
    def __init__(self, hidden_size=768, dropout_r=0.1):
        super(ATT, self).__init__()

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q):

        v = self.linear_v(v)
        k = self.linear_k(k)
        q = self.linear_q(q)

        atted = self.att(v, k, q)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)
        # print(query.size())
        # print(key.size())
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size=768, ff_size=768, dropout_r=0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------------
# ----  Guided Attention ----
# -------------------------------

class GA(nn.Module):
    def __init__(self,hidden_size=768, dropout_r=0.1):
        super(GA, self).__init__()

        self.att1 = ATT(hidden_size, dropout_r)
        self.ffn = FFN(hidden_size, hidden_size, dropout_r)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = Layernorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = Layernorm(hidden_size)

    def forward(self, x, y):
        y = self.norm1(y + self.dropout1(
            self.att1(x,x,y)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y
