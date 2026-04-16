import sys
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Shared building blocks (root src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from transformer_blocks import MyTransformer, SinusoidalPositionalEncoding  # noqa: F401


class CadmusDNA(nn.Module):

    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Carica modello LLM
        self.model_LLM = AutoModel.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-2.5B-multi-species",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.model_LLM.eval()

        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model_LLM(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Embedding: [Batch, Seq_Len, Hidden_Dim]
            embedding_tot = outputs.last_hidden_state

        output_att, importance = self.attention_model(embedding_tot)
        return output_att, importance


class TransformerNuc_Cadmus(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout_rate=0.0,
                 f_activation=nn.ReLU()):
        super(TransformerNuc_Cadmus, self).__init__()

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
        # seq: (batch_size, seq_len, feature_dim)
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0, :]
        out = self.final_linear(out)
        return torch.squeeze(out), attention_matrix
