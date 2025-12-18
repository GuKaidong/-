# models/segrnn_model_grid.py
import torch
import torch.nn as nn

class SegRNNGrid(nn.Module):
    """
    SegRNN adapted to predict grid-index per future timestep/block.
    - segment_w: segment width; use 1 to predict each timestep individually.
    - in_channels: number of features per timestep (e.g., x,y,alt,vel -> 4).
    - num_grids: total classes (grid_size * grid_size).
    """
    def __init__(self, segment_w=1, hidden_d=128, num_grids=16384,
                 gru_layers=1, dropout=0.2, max_m=512, in_channels=2):
        super().__init__()
        self.w = int(segment_w)
        self.d = int(hidden_d)
        self.C = int(in_channels)
        self.num_grids = int(num_grids)

        self.proj = nn.Linear(self.w * self.C, self.d)
        self.proj_act = nn.ReLU()
        self.gru = nn.GRU(input_size=self.d, hidden_size=self.d,
                          num_layers=gru_layers, batch_first=True)
        self.decoder_cell = nn.GRUCell(input_size=self.d, hidden_size=self.d)
        self.rp_emb = nn.Embedding(max_m, self.d)
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(self.d, self.num_grids)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.)

    def forward(self, x: torch.Tensor, target_m: int):
        """
        x: (B, L, C)
        target_m: m (number of prediction blocks)
        returns logits: (B, m, num_grids)
        """
        B, L, C = x.shape
        assert C == self.C, f"Input channel mismatch: {C} vs model {self.C}"

        # 以最后观测点做相对归一（注意：如果你的 X 已经全局标准化，这里相对差仍然有意义）
        last = x[:, -1:, :].clone()  # (B,1,C)
        x_norm = x - last

        assert L % self.w == 0, f"L ({L}) must be divisible by w ({self.w})"
        n = L // self.w
        x_seg = x_norm.view(B, n, self.w * C)           # (B,n,w*C)
        x_proj = self.proj_act(self.proj(x_seg))        # (B,n,d)

        _, h_n = self.gru(x_proj)                       # (num_layers,B,d)
        hn = h_n[-1]                                    # (B,d)

        m = int(target_m)
        idx = torch.arange(m, device=hn.device)
        rp = self.rp_emb(idx)                           # (m,d)
        pe = rp.unsqueeze(0).expand(B, -1, -1)          # (B,m,d)
        hn_rep = hn.unsqueeze(1).expand(-1, m, -1)      # (B,m,d)

        pe_flat = pe.reshape(B * m, self.d)
        hn_flat = hn_rep.reshape(B * m, self.d)

        dec_out_flat = self.decoder_cell(pe_flat, hn_flat)      # (B*m,d)
        dec_out = dec_out_flat.view(B, m, self.d)               # (B,m,d)
        dec_out = self.dropout(dec_out)

        logits_flat = self.pred(dec_out.reshape(B * m, self.d)) # (B*m,num_grids)
        logits = logits_flat.view(B, m, self.num_grids)         # (B,m,num_grids)
        return logits
