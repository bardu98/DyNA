import os
import sys
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('../MAIN PHASE/src'))
from model import CadmusDNA, TransformerNuc_Cadmus

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seeds fixed with seed = {seed}")

# ------------
# SMOOTHING
# ------------
def smooth_signal(probs, window_size):

    if window_size < 2:
        return probs
    s = pd.Series(probs)
    smoothed = s.rolling(window=window_size, center=True, min_periods=1).mean().values
    return smoothed


class SlidingWindowDataset(Dataset):
    def __init__(self, full_seq, window_size=201):
        self.windows = []
        self.centers = []
        self.window_size = window_size
        half_win = window_size // 2
        
        for c in range(half_win, len(full_seq) - half_win):
            start = c - half_win
            end = c + half_win + 1
            window = full_seq[start:end]
            
            if len(window) == window_size:
                #masking
                masked_fw = window[:18] + ('N' * 18) + window[36:162] + ('N' * 18) + window[180:]
                masked_rc = str(Seq(masked_fw).reverse_complement())
                
                self.windows.append({
                    'sequence': masked_fw, 
                    'sequence_rev': masked_rc
                })
                self.centers.append(c)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def output_model_batch_inference(batch, model, tokenizer, device):
    sequences_fw = list(batch['sequence']) if isinstance(batch['sequence'], tuple) else batch['sequence']
    sequences_rc = list(batch['sequence_rev']) if isinstance(batch['sequence_rev'], tuple) else batch['sequence_rev']
        
    tokens_fw = tokenizer(sequences_fw, return_tensors="pt", padding=True, truncation=True)
    input_ids_fw = tokens_fw['input_ids'].to(device)
    attention_mask_fw = tokens_fw['attention_mask'].to(device)
    output_fw, _ = model(input_ids_fw, attention_mask_fw)

    tokens_rc = tokenizer(sequences_rc, return_tensors="pt", padding=True, truncation=True)
    input_ids_rc = tokens_rc['input_ids'].to(device)
    attention_mask_rc = tokens_rc['attention_mask'].to(device)
    output_rc, _ = model(input_ids_rc, attention_mask_rc)

    return output_fw, output_rc

def predict_sliding_windows(model, dataloader, tokenizer, device):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            out_fw, out_rc = output_model_batch_inference(batch, model, tokenizer, device)
            probs = torch.sigmoid((out_fw + out_rc) / 2)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            
    return np.array(all_probs)

# ------
# PLOT 
# ------
def process_and_plot_sequence(row_idx, row_data, model, tokenizer, device, output_dir, batch_size=64, smoothing_window=0):
    full_seq = row_data['sequenza']
    true_scores = np.array(row_data['dyad_scores'])
    chrom = row_data['chr']
    seq_start = row_data['start']
    
    sw_dataset = SlidingWindowDataset(full_seq, window_size=201)
    if len(sw_dataset) == 0:
        print(f"  [!] La sequenza all'indice {row_idx} non è valida per lo sliding window.")
        return
        
    sw_loader = DataLoader(sw_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  [*] Inferenza su {len(sw_dataset)} finestre per seq_idx {row_idx}...")
    preds_probs = predict_sliding_windows(model, sw_loader, tokenizer, device)
    
    # -------
    # apply smoothing
    # -------
    if smoothing_window > 1:
        preds_probs = smooth_signal(preds_probs, smoothing_window)
        pred_label = f'Predicted Probability (Smoothed, w={smoothing_window})'
    else:
        pred_label = 'Predicted Probability (DyNA)'

    valid_centers = sw_dataset.centers
    aligned_true_scores = true_scores[valid_centers]
    
    pearson_corr, _ = pearsonr(aligned_true_scores, preds_probs)
    spearman_corr, _ = spearmanr(aligned_true_scores, preds_probs)
    
    # Plot
    plt.figure(figsize=(14, 6))
    genomic_positions = [seq_start + c for c in valid_centers]
    
    ax1 = plt.gca()
    line1 = ax1.plot(genomic_positions, aligned_true_scores, color='#006B7D', linewidth=2.5, label='True WIG Score')
    ax1.set_xlabel('Genomic Position (bp)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Nucleosome Signal (WIG)', fontsize=14, fontweight='bold', color='#006B7D')
    ax1.tick_params(axis='y', labelcolor='#006B7D')
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(genomic_positions, preds_probs, color='#D95319', linewidth=2.5, alpha=0.85, label=pred_label)
    ax2.set_ylabel('Predicted Probability', fontsize=14, fontweight='bold', color='#D95319')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='#D95319')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    
    title_suffix = f" (Smoothed w={smoothing_window})" if smoothing_window > 1 else ""
    plt.title(f'Continuous Nucleosome Prediction vs Real Signal ({chrom}){title_suffix}\nPearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f}', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"continuous_prediction_seq{row_idx}_{chrom}_{seq_start}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Plot salvato in: {save_path} (Pearson: {pearson_corr:.3f})")

# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description="Sliding Window Inference and Correlation Plot for CadmusDNA")
    parser.add_argument("--dataset", type=str, required=True, help="Path al file dataset (.pkl) con le sequenze da 2000bp")
    parser.add_argument("--weights", type=str, required=True, help="Path al file dei pesi .pt")
    parser.add_argument("--output_dir", type=str, default="./results_continuous", help="Cartella di output per i plot")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per la sliding window")
    parser.add_argument("--num_plots", type=int, default=5, help="Quante sequenze random plottare dal dataset")
    parser.add_argument("--smoothing", type=int, default=0, help="Finestra di smoothing per la predizione (es. 5, 11). 0 = disabilitato.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Caricamento dataset continuo da: {args.dataset}")
    df_continuous = pd.read_pickle(args.dataset)
    print(f"[*] Trovate {len(df_continuous)} sequenze nel dataset.")

    indices_to_plot = random.sample(range(len(df_continuous)), min(args.num_plots, len(df_continuous)))

    print("[*] Inizializzazione modello e caricamento pesi...")
    best_hyperparameters = {'dropout_rate': 0.3143462158665756, 'num_heads': 8}
    transf_parameters_att = {
        'input_dim': 2560, 
        'dropout_rate': best_hyperparameters['dropout_rate'], 
        'num_heads': best_hyperparameters['num_heads']
    }
    
    model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    print("[*] Caricamento Tokenizer...")
    model_name = "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n" + "="*50)
    print(f"ELABORAZIONE E PLOTTING ({len(indices_to_plot)} SEQUENZE)")
    if args.smoothing > 1:
        print(f"[*] ATTENZIONE: Smoothing attivo con finestra = {args.smoothing} bp")
    print("="*50)

    for i, idx in enumerate(indices_to_plot):
        print(f"\n[{i+1}/{len(indices_to_plot)}] Elaborazione riga {idx} del dataframe...")
        row_data = df_continuous.iloc[idx]
        process_and_plot_sequence(
            row_idx=idx, 
            row_data=row_data, 
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            smoothing_window=args.smoothing
        )
        
    print("\n[*] Processo completato con successo!")

if __name__ == "__main__":
    main()