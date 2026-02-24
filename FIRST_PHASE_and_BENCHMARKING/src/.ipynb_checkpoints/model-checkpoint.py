import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # # seq: (batch_size, seq_len, feature_dim) → no need to transpose
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0,:]
        out = self.final_linear(out)

        return torch.squeeze(out), attention_matrix


class CadmusDNA(nn.Module):
    
    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        # Imposta il dispositivo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
        # # Carica modello e tokenizer una sola volta
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
        # )
        # self.model_LLM = AutoModel.from_pretrained(
        #     "InstaDeepAI/nucleotide-transformer-2.5B-multi-species",
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        # ).to(self.device)
        # self.model_LLM.eval()
    
        # Inizializza i moduli
        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)
        

    # def batch_embeddings(self, sequences, batch_size=32):
    #     """Calcola gli embedding completi per una lista di sequenze in batch."""
    #     all_embeddings = []
    #     for i in tqdm(range(0, len(sequences), batch_size), disable=True):
    #         batch = sequences[i:i+batch_size]
    #         tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model_LLM(**tokens)
    #             # Mantiene tutti gli hidden states (nessun pooling)
    #             batch_emb = outputs.last_hidden_state.cpu()
    #         all_embeddings.extend(batch_emb)
    #     return all_embeddings
        

    def forward(self, seqs):
        #print(seqs)
        #embedding_tot = torch.stack(self.batch_embeddings(seqs)).to(self.device)
        output_att, importance = self.attention_model(seqs)
        return output_att, importance



class MyTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate, activation=nn.ReLU()):
        super(MyTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.act = activation
        self.positional_encoding = SinusoidalPositionalEncoding(self.embedding_dim, max_len=73)

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
        # seq: (batch_size, seq_len, embedding_dim)
        seq = self.positional_encoding(seq)
        attn_output, attention_matrix = self.multihead_attention(seq, seq, seq)
        attn_output = self.norm1(attn_output + seq)
        ffn_out = self.pw_ffnn(attn_output)
        out = self.norm2(attn_output + ffn_out)
        return out, attention_matrix


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=28):    

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
