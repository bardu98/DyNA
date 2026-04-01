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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Assicurati che il path punti alla tua cartella corretta con il modello
sys.path.append(os.path.abspath('../src'))
from model import CadmusDNA, TransformerNuc_Cadmus

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO E DATASET
# -------------------------------------------------------------------------
def smooth_signal(probs, window_size=11):
    """Applica una media mobile centrata per trovare il 'centro di massa' del picco."""
    if window_size < 2:
        return probs
    s = pd.Series(probs)
    return s.rolling(window=window_size, center=True, min_periods=1).mean().values

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
                # Structural Masking
                masked_fw = window[:18] + ('N' * 18) + window[36:162] + ('N' * 18) + window[180:]
                masked_rc = str(Seq(masked_fw).reverse_complement())
                self.windows.append({'sequence': masked_fw, 'sequence_rev': masked_rc})
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

# -------------------------------------------------------------------------
# CALCOLO OFFSET
# -------------------------------------------------------------------------
def calculate_offsets(df, model, tokenizer, device, num_samples=500, batch_size=128):
    """Calcola la distanza in bp tra il picco predetto e il picco reale."""
    offsets = []
    
    # Seleziona un sottoinsieme casuale per non far durare l'inferenza ore
    indices = random.sample(range(len(df)), min(num_samples, len(df)))
    
    print(f"\n[*] Calcolo Offset su {len(indices)} sequenze...")
    
    for i, idx in enumerate(tqdm(indices, desc="Analisi Sequenze")):
        row_data = df.iloc[idx]
        full_seq = row_data['sequenza']
        true_scores = np.array(row_data['dyad_scores'])
        
        sw_dataset = SlidingWindowDataset(full_seq, window_size=201)
        if len(sw_dataset) == 0: continue
            
        sw_loader = DataLoader(sw_dataset, batch_size=batch_size, shuffle=False)
        preds_probs = predict_sliding_windows(model, sw_loader, tokenizer, device)
        
        # Allineamento
        valid_centers = sw_dataset.centers
        aligned_true_scores = true_scores[valid_centers]
        
        # Smoothing (11 bp = ~1 giro d'elica) per trovare il centro di massa del nucleosoma
        smooth_true = smooth_signal(aligned_true_scores, window_size=11)
        smooth_pred = smooth_signal(preds_probs, window_size=11)
        
        # Trova gli indici dei picchi massimi
        true_peak_idx = np.argmax(smooth_true)
        pred_peak_idx = np.argmax(smooth_pred)
        
        # Calcola l'offset (Posizione Predetta - Posizione Reale)
        # Se offset < 0, il modello ha predetto il picco "prima" del reale.
        # Se offset > 0, il modello ha predetto il picco "dopo" il reale.
        offset = pred_peak_idx - true_peak_idx
        offsets.append(offset)
        
    return np.array(offsets)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculate Nucleosome Peak Offset Distribution")
    parser.add_argument("--dataset", type=str, required=True, help="Path al file dataset (.pkl) con le sequenze da 2000bp")
    parser.add_argument("--weights", type=str, required=True, help="Path al file dei pesi .pt")
    parser.add_argument("--output_dir", type=str, default="./results_continuous", help="Cartella di output per il plot")
    parser.add_argument("--num_samples", type=int, default=500, help="Numero di sequenze su cui calcolare l'offset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Caricamento dataset continuo da: {args.dataset}")
    df_continuous = pd.read_pickle(args.dataset)

    print("[*] Inizializzazione modello e tokenizer...")
    transf_parameters_att = {'input_dim': 2560, 'dropout_rate': 0.3143462158665756, 'num_heads': 8}
    model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5B-multi-species")

    # Esecuzione del calcolo degli offset
    offsets = calculate_offsets(df_continuous, model, tokenizer, device, num_samples=args.num_samples)
    
    if len(offsets) == 0:
        print("[!] Nessun offset calcolato.")
        return

    # -------------------------------------------------------------------------
    # CREAZIONE DEL PLOT DELLA DISTRIBUZIONE (KDE + Istogramma)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    # Istogramma con curva KDE
    sns.histplot(offsets, bins=range(-150, 150, 5), kde=True, color='#006B7D', stat='density', alpha=0.6)
    
    # Linee di riferimento
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Match (0 bp)')
    
    # Calcolo metriche per il plot
    median_offset = np.median(offsets)
    mean_offset = np.mean(offsets)
    std_offset = np.std(offsets)
    
    # Percentuali di accuratezza spaziale (Molto usate nei paper!)
    within_10bp = np.mean(np.abs(offsets) <= 10) * 100
    within_20bp = np.mean(np.abs(offsets) <= 20) * 100
    within_50bp = np.mean(np.abs(offsets) <= 50) * 100

    # Aggiungi un box di testo con le statistiche
    stats_text = (
        f"Statistiche su {len(offsets)} picchi:\n"
        f"Mean Offset: {mean_offset:.1f} bp\n"
        f"Median Offset: {median_offset:.1f} bp\n"
        f"Std Dev: {std_offset:.1f} bp\n\n"
        f"Resolution Accuracy:\n"
        f"Within $\pm$10 bp: {within_10bp:.1f}%\n"
        f"Within $\pm$20 bp: {within_20bp:.1f}%\n"
        f"Within $\pm$50 bp: {within_50bp:.1f}%"
    )
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

    plt.title('Nucleosome Prediction Resolution (Offset Distribution)', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Distance from True Peak (Base Pairs)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.xlim(-150, 150)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, "offset_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[✓] Plot della distribuzione dell'offset salvato in: {save_path}")
    
    # Stampa in console
    print("\n" + "="*50)
    print("RISULTATI DI RISOLUZIONE SPAZIALE (OFFSET)")
    print("="*50)
    print(stats_text)

if __name__ == "__main__":
    main()