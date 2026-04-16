import sys
import os

import torch
import torch.nn as nn

# Shared building blocks (root src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from transformer_blocks import MyTransformer, SinusoidalPositionalEncoding  # noqa: F401


class TransformerNuc_DyNA(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout_rate=0.0,
                 f_activation=nn.ReLU()):
        super(TransformerNuc_DyNA, self).__init__()

        self.transoformer = MyTransformer(
            embedding_dim=input_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            activation=f_activation
        )
        self.act = f_activation

        self.final_ffn = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

        self.final_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 1),
        )

    def forward(self, seq):
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0, :]
        out = self.final_linear(out)
        return torch.squeeze(out), attention_matrix


class DyNA(nn.Module):

    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)

    def forward(self, seqs):
        output_att, importance = self.attention_model(seqs)
        return output_att, importance
