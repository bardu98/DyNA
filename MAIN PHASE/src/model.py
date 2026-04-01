
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CadmusDNA(nn.Module):
    
    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        # Imposta il dispositivo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Carica modello LLM
        self.model_LLM = AutoModel.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-2.5B-multi-species",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.model_LLM.eval()
        
        # Inizializza i moduli
        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)
        
    def forward(self, input_ids, attention_mask):
        """
        Nuovo forward ottimizzato per ricevere tensori.
        """
        # 1. LLM Embedding (Zero copie CPU-GPU)
        with torch.no_grad():
            outputs = self.model_LLM(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            # Embedding: [Batch, Seq_Len, Hidden_Dim]
            embedding_tot = outputs.last_hidden_state
        
        # 2. Attention Head
        output_att, importance = self.attention_model(embedding_tot)
        
        return output_att, importance


class TransformerNuc_Cadmus(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout_rate=0.0, 
                 f_activation=nn.ReLU()):
        super(TransformerNuc_Cadmus, self).__init__()

        self.transoformer = MyTransformer(embedding_dim=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, activation=f_activation)
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
        out = out[:, 0,:]
        out = self.final_linear(out)

        return torch.squeeze(out), attention_matrix


class MyTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate, activation=nn.ReLU()):
        super(MyTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.act = activation
        
        # Impostato max_len a 5000 per sicurezza
        self.positional_encoding = SinusoidalPositionalEncoding(self.embedding_dim, max_len=5000)    

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)

        self.pw_ffnn = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.embedding_dim)
        )

    def forward(self, seq):
        seq = self.positional_encoding(seq)
        attn_output, attention_matrix = self.multihead_attention(seq, seq, seq)
        attn_output = self.norm1(attn_output + seq)
        ffn_out = self.pw_ffnn(attn_output)
        out = self.norm2(attn_output + ffn_out)
        return out, attention_matrix


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000): 
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]






