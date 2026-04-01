#!/usr/bin/env python3
import os
import pickle
import argparse
import random
import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from transformers import AutoTokenizer

# ==============================================================================
# 1. CONFIGURAZIONI GLOBALI
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.dpi': 300,
    'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False
})

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. CARICAMENTO DATI E MAPPING 
# ==============================================================================
def map_attention_to_bp_static(imp_array, input_ids_tensor, tokenizer, seq_len=201):
    if imp_array.ndim > 2:
        imp_array = np.mean(imp_array, axis=1)
    batch_size = imp_array.shape[0]
    bp_att = np.zeros((batch_size, seq_len))
    input_ids_list = input_ids_tensor.cpu().numpy()
    
    for i in tqdm(range(batch_size), desc="Mapping Attention to BP", leave=False):
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list[i])
        seq_idx = 0
        for t_idx, token in enumerate(tokens):
            if token in ['[CLS]', '<cls>', '<s>', '[PAD]', '<pad>', '[SEP]', '</s>']: continue
            clean_token = token.replace(' ', '').replace('Ġ', '')
            if clean_token in ['[UNK]', '<unk>']: clean_token = 'N' 
            t_len = len(clean_token)
            if t_len == 0: continue
            
            end_idx = min(seq_idx + t_len, seq_len)
            actual_len = end_idx - seq_idx
            if actual_len > 0:
                bp_att[i, seq_idx : end_idx] = imp_array[i, t_idx]
                seq_idx = end_idx
            if seq_idx >= seq_len: break
    return bp_att

def process_matrices(dataset, avg_dir, avg_rc):
    print("[*] Avvio Base-Pair mapping per normalizzare le N...")
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5B-multi-species")
    seqs_fw = [d['sequence'] for d in dataset]
    seqs_rc = [d['sequence_rev'] for d in dataset]

    tok_fw = tokenizer(seqs_fw, padding=True, truncation=True, return_tensors='pt')
    tok_rc = tokenizer(seqs_rc, padding=True, truncation=True, return_tensors='pt')

    bp_fw = map_attention_to_bp_static(avg_dir.numpy(), tok_fw['input_ids'], tokenizer)
    bp_rc = map_attention_to_bp_static(avg_rc.numpy(), tok_rc['input_ids'], tokenizer)

    bp_rc_flipped = np.flip(bp_rc, axis=1)
    sym_bp_att = (bp_fw + bp_rc_flipped) / 2.0

    print("[*] Ri-impacchettamento in 34 blocchi da 6-bp...")
    N_samples = sym_bp_att.shape[0]
    k = 6
    num_bins = int(np.ceil(201 / k))
    binned_att = np.zeros((N_samples, num_bins))

    for i in range(N_samples):
        for b in range(num_bins):
            start = b * k
            end = min((b + 1) * k, 201)
            binned_att[i, b] = np.mean(sym_bp_att[i, start:end])

    return binned_att

def applica_mascheramento(dataset):
    for d in dataset:
        seq = d['sequence']
        masked_seq = seq[:18] + ('N' * 18) + seq[36:162] + ('N' * 18) + seq[180:]
        d['sequence'] = masked_seq
    return dataset

def load_data_and_matrices(args):
    print(f"[*] Caricamento dataset: {args.cell_type.upper()}")
    ds_map = {'lympho': 'Lymphoblastoid_99_8_percentile.pkl', 'act': 'CD4T_h19_Act_tot_99_8_percentile.pkl', 'rest': 'CD4T_h19_Rest_tot_99_8_percentile.pkl'}
    pred_map = {'lympho': 'preds_lymphoblastoid_model_sum_folds_08_03_26_MASK.pkl', 'act': 'preds_act_model_sum_folds_08_03_26_MASK.pkl', 'rest': 'preds_rest_model_sum_folds_08_03_26_MASK.pkl'}
    mat_type = 'lymp' if args.cell_type == 'lympho' else args.cell_type

    with open(os.path.join(args.data_dir, ds_map[args.cell_type]), 'rb') as f: 
        dataset = pickle.load(f)

    dataset = applica_mascheramento(dataset) 
    for d in dataset: 
        d['sequence_rev'] = str(Seq(d['sequence']).reverse_complement())

    total_matrices_dir, total_matrices_rc = None, None

    for fold in range(5):
        file_path = os.path.join(args.results_dir, f"matrices_results_fold{fold}_{mat_type}_MASK_08_02_2026.pt")
        data = torch.load(file_path, map_location='cpu')

        curr_dir, curr_rc = torch.stack(data['matrices_dir']), torch.stack(data['matrices_rc'])
        if fold == 0: 
            total_matrices_dir, total_matrices_rc = curr_dir, curr_rc
        else: 
            total_matrices_dir += curr_dir; total_matrices_rc += curr_rc
            
    avg_dir, avg_rc = total_matrices_dir / 5, total_matrices_rc / 5
    
    preds_df = pd.read_pickle(os.path.join(args.results_dir, pred_map[args.cell_type]))
    preds_int = (np.array(preds_df) > 0.5).astype(int).flatten()

    binned_att = process_matrices(dataset, avg_dir, avg_rc)
    return dataset, binned_att, preds_int

# ==============================================================================
# 3. FFT ANALYSIS (SINGLE SEQUENCE AGGREGATION)
# ==============================================================================
def plot_attention_periodicity(binned_att, preds_int, val_labels, args):
    print(f"[*] Generazione Single-Sequence FFT Plot per {args.cell_type.upper()}...")

    mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    mask_tn = [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)]
    
    tp_attentions = binned_att[torch.tensor(mask_tp, dtype=torch.bool)]
    tn_attentions = binned_att[torch.tensor(mask_tn, dtype=torch.bool)]

    print(f"[*] Elaborazione FFT indipendente su {tp_attentions.shape[0]} TP e {tn_attentions.shape[0]} TN.")

    if tp_attentions.shape[0] < 5 or tn_attentions.shape[0] < 5:
        print("[!] Troppi pochi campioni per eseguire l'analisi FFT.")
        return

    # --- TAGLIO DEI BORDI MASCHERATI ---
    # Analizziamo SOLO i bin intatti per evitare artefatti a bassa frequenza dovuti alle N.
    # Bin 6 (bp 36) a Bin 26 incluso (bp 162) = 21 bin = 126 bp
    CORE_START_BIN = 6
    CORE_END_BIN = 27
    
    tp_core = tp_attentions[:, CORE_START_BIN:CORE_END_BIN]
    tn_core = tn_attentions[:, CORE_START_BIN:CORE_END_BIN]

    # --- CALCOLO FFT PER OGNI SINGOLA SEQUENZA ---
    # fs = 1/6 campioni per bp. 'axis=1' applica la FFT riga per riga, in modo indipendente.
    # 'detrend=constant' rimuove la media per ogni singola riga prima della FFT.
    fs = 1.0 / 6.0 
    freqs, psd_tp_all = periodogram(tp_core, fs=fs, axis=1, detrend='constant')
    _, psd_tn_all = periodogram(tn_core, fs=fs, axis=1, detrend='constant')

    # Ignoriamo la frequenza 0
    valid_idx = freqs > 0
    freqs = freqs[valid_idx]
    periods = 1 / freqs
    
    psd_tp_all = psd_tp_all[:, valid_idx]
    psd_tn_all = psd_tn_all[:, valid_idx]

    # --- AGGREGAZIONE (Media e Standard Error) ---
    mean_psd_tp = np.mean(psd_tp_all, axis=0)
    sem_psd_tp = np.std(psd_tp_all, axis=0) / np.sqrt(psd_tp_all.shape[0])
    
    mean_psd_tn = np.mean(psd_tn_all, axis=0)
    sem_psd_tn = np.std(psd_tn_all, axis=0) / np.sqrt(psd_tn_all.shape[0])

    # Troviamo l'indice del periodo più vicino all'aliasing atteso (~14.2 bp)
    target_period = 14.2
    idx_target = (np.abs(periods - target_period)).argmin()
    actual_target_period = periods[idx_target]

    # --- PLOTTING MULTIPANNELLO PER IL PAPER ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})
    sns.set_style("whitegrid", {"axes.facecolor": '#FDF7F7', "grid.color": "#e0e0e0"})

    # PANNELLO 1: Spettro di Potenza Medio
    ax1.plot(periods, mean_psd_tp, color='#D4880F', linewidth=3, marker='o', markersize=7, label='True Positives')
    ax1.fill_between(periods, mean_psd_tp - sem_psd_tp, mean_psd_tp + sem_psd_tp, color='#D4880F', alpha=0.2)
    
    ax1.plot(periods, mean_psd_tn, color='#006B7D', linewidth=3, marker='o', markersize=7, label='True Negatives')
    ax1.fill_between(periods, mean_psd_tn - sem_psd_tn, mean_psd_tn + sem_psd_tn, color='#006B7D', alpha=0.2)

    ax1.axvline(12.0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Nyquist Limit (12 bp)')
    ax1.axvspan(13.5, 15.0, color='red', alpha=0.15, label=f'Expected Aliasing (~14.2 bp)')
    
    ax1.set_xlabel('Periodicity (Base Pairs)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Mean Power Spectral Density', fontsize=16, fontweight='bold')
    ax1.set_title('Aggregate Single-Sequence FFT Spectrum', fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlim(max(periods) + 1, 11) # Invertiamo l'asse
    ax1.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='gray')

    # PANNELLO 2: Distribuzione Statistica al Picco (Violin Plot)
    # Estraiamo la potenza di tutte le sequenze a quella specifica frequenza per il violin plot
    tp_target_power = psd_tp_all[:, idx_target]
    tn_target_power = psd_tn_all[:, idx_target]

    df_viol = pd.DataFrame({
        'Power': np.concatenate([tp_target_power, tn_target_power]),
        'Group': ['True Positives\n(Nucleosomes)'] * len(tp_target_power) + ['True Negatives\n(Linkers)'] * len(tn_target_power)
    })

    sns.violinplot(data=df_viol, x='Group', y='Power', palette={'True Positives\n(Nucleosomes)': '#D4880F', 'True Negatives\n(Linkers)': '#006B7D'}, 
                   ax=ax2, inner="quartile", cut=0)
    
    # Test Statistico: i TP hanno una potenza maggiore dei TN a questa frequenza?
    _, p_val = mannwhitneyu(tp_target_power, tn_target_power, alternative='greater')
    
    y_max = df_viol['Power'].quantile(0.95) # Limitiamo l'asse Y per evitare che outliers estremi schiaccino il violino
    ax2.set_ylim(bottom=0, top=y_max * 1.3)
    
    # Barra di significatività
    h = y_max * 1.15
    ax2.plot([0, 0, 1, 1], [h, h*1.02, h*1.02, h], lw=1.5, c='k')
    ax2.text(0.5, h*1.03, "p < 0.001" if p_val < 0.001 else f"p = {p_val:.2e}", ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    ax2.set_title(f'Signal Power at ~{actual_target_period:.1f} bp', fontsize=18, fontweight='bold', pad=15)
    ax2.set_ylabel('Power Spectral Density', fontsize=14, fontweight='bold')
    ax2.set_xlabel('')

    plt.suptitle(f'Single-Sequence Attention Periodicity ({args.cell_type.upper()})', fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()

    save_path = os.path.join(args.out_dir, f'attention_fft_per_sequence_{args.cell_type}_MASK.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot Aggregato salvato in: {save_path}")

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Standalone Tool - Single-Sequence Attention Periodicity FFT")
    parser.add_argument("--data_dir", type=str, default="../data/data_pkl", help="Cartella dei dataset .pkl")
    parser.add_argument("--results_dir", type=str, default="../results", help="Cartella dei tensori .pt e predizioni")
    parser.add_argument("--out_dir", type=str, default="../images", help="Cartella per salvare l'immagine")
    parser.add_argument("--cell_type", type=str, choices=['lympho', 'act', 'rest'], required=True)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_all, binned_att, preds_int = load_data_and_matrices(args)
    val_labels = [s['label'] for s in dataset_all]

    plot_attention_periodicity(binned_att, preds_int, val_labels, args)

if __name__ == "__main__":
    main()