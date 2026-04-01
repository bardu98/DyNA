import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from matplotlib.lines import Line2D

# Aggiusta il path se necessario
sys.path.append(os.path.abspath('../MAIN PHASE/src'))
from model import CadmusDNA, TransformerNuc_Cadmus
from data_class import Nuc_Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seeds fixed with seed = {seed}")

class InterpretableCadmus(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model_LLM = original_model.model_LLM
        self.attention_model = original_model.attention_model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model_LLM(input_ids=input_ids, attention_mask=attention_mask)
        logit, _ = self.attention_model(outputs.last_hidden_state) 
        return logit.view(-1, 1)

# =========================================================================
# CALCOLO GLOBALE ALLINEATO AL NUCLEOTIDE
# =========================================================================
def compute_global_importance_RC(dataloader, lig, device):
    """
    Accumula IG calcolando l'allineamento perfetto a livello di paia di basi (bp).
    Risolve il problema dello sfasamento causato dai k-mers.
    """
    print(f"[*] Inizio calcolo Global Importance Profile su {len(dataloader.dataset)} sequenze...")
    
    seq_len_bp = 201
    global_bp_sum = np.zeros(seq_len_bp)
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Processing Batches (IG)"):
        input_ids_fwd = batch['input_ids'].to(device)
        mask_fwd = batch['attention_mask'].to(device)
        input_ids_rc = batch['input_ids_rc'].to(device)
        mask_rc = batch['attention_mask_rc'].to(device)

        current_batch_size = input_ids_fwd.size(0)
        
        # --- CALCOLO IG ---
        attr_fwd = lig.attribute(
            inputs=input_ids_fwd, additional_forward_args=(mask_fwd,),
            target=0, n_steps=20, internal_batch_size=4
        )
        sum_fwd = attr_fwd.sum(dim=-1).detach().cpu().numpy() # [Batch, 37]
        
        attr_rc = lig.attribute(
            inputs=input_ids_rc, additional_forward_args=(mask_rc,),
            target=0, n_steps=20, internal_batch_size=4
        )
        sum_rc = attr_rc.sum(dim=-1).detach().cpu().numpy() # [Batch, 37]

        # --- ALLINEAMENTO BASE-PAIR PER OGNI SEQUENZA ---
        for i in range(current_batch_size):
            # 1. Prendiamo solo i 33 k-mer centrali
            fwd_tokens = sum_fwd[i][1:-3] 
            rc_tokens = sum_rc[i][1:-3]   

            # 2. Espandiamo a nucleotidi (33 * 6 = 198 bp)
            fwd_bp = np.repeat(fwd_tokens, 6)
            rc_bp = np.repeat(rc_tokens, 6)

            # 3. Il RC deve essere ribaltato per allinearsi geometricamente al Forward
            rc_bp_aligned = rc_bp[::-1]

            # 4. Posizionamento sulla sequenza esatta di 201 bp
            seq_imp = np.zeros(seq_len_bp)

            # FWD copre i primi 198 nucleotidi (da 0 a 197)
            seq_imp[0:198] += fwd_bp
            # RC ribaltato copre gli ultimi 198 nucleotidi (da 3 a 200)
            seq_imp[3:201] += rc_bp_aligned

            global_bp_sum += seq_imp

        total_samples += current_batch_size
        
    # Calcolo della media esatta dividendo correttamente le zone di overlap
    overlap_counts = np.zeros(seq_len_bp)
    overlap_counts[0:3] = 1        # Solo FWD presente qui
    overlap_counts[3:198] = 2      # Sia FWD che RC si sovrappongono
    overlap_counts[198:201] = 1    # Solo RC presente qui

    total_counts = overlap_counts * total_samples
    global_avg_bp = global_bp_sum / total_counts
    
    return global_avg_bp

# =========================================================================
# PLOT METAPROFILO CON SIMMETRIZZAZIONE
# =========================================================================
def plot_metaprofile(avg_bp_scores, output_dir, dataset_size):
    # 1. Normalizzazione (Mantiene il segno)
    max_val = np.max(np.abs(avg_bp_scores))
    if max_val > 0:
        norm_bp_scores = avg_bp_scores / max_val
    else:
        norm_bp_scores = avg_bp_scores

    # 2. SIMMETRIZZAZIONE STRUTTURALE (GOLD STANDARD)
    # Forza la simmetria perfetta matematica attorno alla diade
    sym_bp_scores = (norm_bp_scores + norm_bp_scores[::-1]) / 2.0

    seq_len_bp = len(sym_bp_scores)

    # 3. Generazione Plot
    plt.figure(figsize=(12, 5))
    x_vals = np.arange(seq_len_bp)

    # Area Positiva (Oro - Nucleosoma)
    plt.fill_between(x_vals, sym_bp_scores, 0, where=(sym_bp_scores >= 0),
                     interpolate=True, color='#F59E0B', alpha=0.9, label='Pro-Nucleosome')

    # Area Negativa (Teal - Linker)
    plt.fill_between(x_vals, sym_bp_scores, 0, where=(sym_bp_scores <= 0),
                     interpolate=True, color='#06B6D4', alpha=0.9, label='Anti-Nucleosome')

    plt.axhline(0, color='#666', lw=1, ls='-')

    plt.title(f"Global Nucleosome Metaprofile (Avg IG over {dataset_size} samples)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Avg Contribution (Normalized)", fontsize=12, fontweight='bold', color='#4B5563')
    plt.xlabel("Position relative to Dyad (bp)", fontsize=12, fontweight='bold', color='#4B5563')

    # Ticks centrati sul Dyad
    dyad_pos = seq_len_bp // 2  # Centro esatto della sequenza (bp 100)
    ticks = [0, dyad_pos-73, dyad_pos, dyad_pos+73, seq_len_bp-1]
    labels = ['Start', '-73', 'Dyad (0)', '+73', 'End']
    
    valid_idx = [i for i, t in enumerate(ticks) if 0 <= t < seq_len_bp]
    plt.xticks([ticks[i] for i in valid_idx], [labels[i] for i in valid_idx])

    sns.despine(left=True)
    
    legend_elements = [
        Line2D([0], [0], color='#F59E0B', lw=4, label='Avg Attraction (Pro-Nuc)'),
        Line2D([0], [0], color='#06B6D4', lw=4, label='Avg Repulsion (Anti-Nuc)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "metaprofile_integrated_gradients_symmetric_MASK.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[*] Plot Metaprofilo salvato in: {save_path}")



def applica_mascheramento(dataset):
    """Sostituisce con 'N' gli indici [18:36] e [162:180]"""
    for d in dataset:
        seq = d['sequence']
        masked_seq = seq[:18] + ('N' * 18) + seq[36:162] + ('N' * 18) + seq[180:]
        d['sequence'] = masked_seq
    return dataset

    

def main():
    parser = argparse.ArgumentParser(description="Metaprofile Importance via Integrated Gradients")
    parser.add_argument("--dataset", type=str, required=True, help="Path al file del dataset (pickle)")
    parser.add_argument("--weights", type=str, required=True, help="Path al file dei pesi del modello (.pt)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Cartella di destinazione")
    parser.add_argument("--sample_size", type=int, default=200, help="Quanti sample usare per l'analisi")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per velocizzare il calcolo")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Caricamento intero dataset da: {args.dataset}")
    dataset_full = pd.read_pickle(args.dataset)

    dataset_full = applica_mascheramento(dataset_full)  #MASKERA !!!!!!!!!!!!!!

    
    actual_size = min(args.sample_size, len(dataset_full))
    print(f"[*] Creazione small_dataset con {actual_size} campioni casuali...")
    random_indices = random.sample(range(len(dataset_full)), actual_size)
    small_dataset = [dataset_full[i] for i in random_indices]

    for d in small_dataset:
        if 'sequence_rev' not in d:
            d['sequence_rev'] = str(Seq(d['sequence']).reverse_complement())

    dataset_obj = Nuc_Dataset(small_dataset, max_length=37, rc_augmentation=True)
    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

    print("[*] Inizializzazione modello...")
    best_hyperparameters = {'dropout_rate': 0.3143462158665756, 'num_heads': 8}
    transf_parameters_att = {
        'input_dim': 2560, 
        'dropout_rate': best_hyperparameters['dropout_rate'], 
        'num_heads': best_hyperparameters['num_heads']
    }
    
    model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    interpretable_model = InterpretableCadmus(model).to(device)
    lig = LayerIntegratedGradients(interpretable_model, interpretable_model.model_LLM.embeddings.word_embeddings)

    avg_bp_scores = compute_global_importance_RC(dataloader, lig, device)

    print("[*] Generazione del Metaprofilo Simmetrico in corso...")
    plot_metaprofile(avg_bp_scores, args.output_dir, actual_size)

if __name__ == "__main__":
    main()