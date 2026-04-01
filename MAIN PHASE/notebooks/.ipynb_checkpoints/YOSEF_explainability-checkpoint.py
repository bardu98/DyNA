#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import random
from collections import defaultdict
from itertools import compress

import pandas as pd
import numpy as np
import torch
from Bio.Seq import Seq
from Bio import motifs
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURAZIONI GLOBALI E DIZIONARI BIOFISICI
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.dpi': 300,
    'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False
})

PROP_TWIST_DIPRO = {'AA': -17.3, 'AC': -6.7, 'AG': -14.3, 'AT': -16.9, 'CA': -8.6, 'CC': -12.8, 'CG': -11.2, 'CT': -14.3, 'GA': -15.1, 'GC': -11.7, 'GG': -12.8, 'GT': -6.7, 'TA': -11.1, 'TC': -15.1, 'TG': -8.6, 'TT': -17.3}
BENDABILITY = {'AA': 3.07, 'AC': 2.97, 'AG': 2.31, 'AT': 2.60, 'CA': 3.58, 'CC': 2.16, 'CG': 2.81, 'CT': 2.31, 'GA': 2.51, 'GC': 3.06, 'GG': 2.16, 'GT': 2.97, 'TA': 6.74, 'TC': 2.51, 'TG': 3.58, 'TT': 3.07}
ROLL_DIPRO = {'AA': 0.7, 'AC': 0.7, 'AG': 4.5, 'AT': 1.1, 'CA': 4.7, 'CC': 3.6, 'CG': 5.4, 'CT': 4.5, 'GA': 1.9, 'GC': 0.3, 'GG': 3.6, 'GT': 0.7, 'TA': 3.3, 'TC': 1.9, 'TG': 4.7, 'TT': 0.7}
MGW = {'AA': 5.30, 'AC': 6.04, 'AG': 5.19, 'AT': 5.31, 'CA': 4.79, 'CC': 4.62, 'CG': 5.16, 'CT': 5.19, 'GA': 4.71, 'GC': 4.74, 'GG': 4.62, 'GT': 6.04, 'TA': 6.4, 'TC': 4.71, 'TG': 4.79, 'TT': 5.30}
HELICAL_TWIST = {'AA': 35.1, 'AC': 31.5, 'AG': 31.9, 'AT': 29.3, 'CA': 37.3, 'CC': 32.9, 'CG': 36.1, 'CT': 31.9, 'GA': 36.3, 'GC': 33.6, 'GG': 32.9, 'GT': 31.5, 'TA': 37.8, 'TC': 36.3, 'TG': 37.3, 'TT': 35.1}
MAJOR_GW = {'AA': 12.15, 'AC': 12.37, 'AG': 13.51, 'AT': 12.87, 'CA': 13.58, 'CC': 15.49, 'CG': 14.42, 'CT': 13.51, 'GA': 13.93, 'GC': 14.55, 'GG': 15.49, 'GT': 12.37, 'TA': 12.32, 'TC': 13.93, 'TG': 13.58, 'TT': 12.15}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seeds fixed (seed = {seed})")

def tokenize_dna_sequence(seq, k=6, max_tokens=33):
    return [seq[i:i + k] for i in range(0, k * max_tokens, k)]

def calculate_physics_complete(seq):
    seq = seq.upper()
    steps_di = [seq[i:i+2] for i in range(len(seq)-1)]
    return {
        'Propeller Twist': np.mean([PROP_TWIST_DIPRO.get(s, 0) for s in steps_di]),
        'Static Bending': np.mean([BENDABILITY.get(s, 0) for s in steps_di]),
        'DNA Roll': np.mean([ROLL_DIPRO.get(s, 0) for s in steps_di]),
        'Minor Groove Width': np.mean([MGW.get(s, 0) for s in steps_di]),
        'Helical Twist': np.mean([HELICAL_TWIST.get(s, 0) for s in steps_di]),
        'Major Groove Width': np.mean([MAJOR_GW.get(s, 0) for s in steps_di])
    }

def get_valid_indices(region):
    if region == 'dyad': return [17]
    elif region == 'shoulder': return [13, 21]
    elif region == 'boundary': return [5, 29]
    elif region == 'global': 
        maske_region = [3, 4, 5, 27, 28, 29]
        all_tokens = list(range(1, 34))
        return [x for x in all_tokens if x not in maske_region]



def smooth_profile(profile, window=5):
    """Applica una media mobile per smussare il rumore dinucleotidico e rivelare l'onda strutturale."""
    return np.convolve(profile, np.ones(window)/window, mode='valid')

def get_sequence_shape_profile(seq, metric_dict):
    """Calcola il profilo della DNA shape step-by-step per l'intera sequenza."""
    seq = seq.upper()
    steps = [seq[i:i+2] for i in range(len(seq)-1)]
    # Usiamo np.nan per i dinucleotidi con 'N', così da ignorarli nel calcolo della media
    return np.array([metric_dict.get(s, np.nan) for s in steps])


# ==============================================================================
# 3. EXTRACTION FUNCTIONS (PHYSICS & MOTIFS)
# ==============================================================================
def extract_top_kmers(mask, dataset, mat_dir, mat_rc, args, top_k=100):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    data_filtered = list(compress(dataset, mask))
    matrices_filt_dir = mat_dir[mask_tensor]
    matrices_filt_rc = mat_rc[mask_tensor]
    
    token_attn, token_counts = defaultdict(float), defaultdict(int)
    valid_indices = get_valid_indices(args.region)

    for is_rc, tensor in [(False, matrices_filt_dir), (True, matrices_filt_rc)]:
        for i, sample in enumerate(data_filtered):
            seq = sample['sequence']
            if is_rc: seq = str(Seq(seq).reverse_complement())
            tokens = tokenize_dna_sequence(seq)
            attn_row = tensor[i][0, :] 
            
            for attn_idx in valid_indices:
                token_pos = attn_idx - 1
                if 0 <= token_pos < len(tokens):
                    t = tokens[token_pos]
                    val = attn_row[attn_idx].item()
                    token_attn[t] += val
                    token_counts[t] += 1

    mean_attn = {k: token_attn[k]/token_counts[k] for k in token_attn}
    sym_attn = {}
    used = set()
    
    for k, v in mean_attn.items():
        if k in used: continue
        k_rc = str(Seq(k).reverse_complement())
        if k_rc in mean_attn:
            score = mean_attn[k] + mean_attn[k_rc] if k != k_rc else mean_attn[k]
            key = '\n'.join(sorted([k, k_rc]))
            used.update([k, k_rc])
        else:
            score, key = v, k
            used.add(k)
        sym_attn[key] = score

    sorted_items = sorted(sym_attn.items(), key=lambda x: x[1], reverse=True)
    top_items_raw = [i[0] for i in sorted_items[:top_k]]
    return [seq for pair in top_items_raw for seq in pair.split('\n')]

def extract_attended_regions(mask, dataset, mat_dir, mat_rc, top_k=100, context_flank=6):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    data_filtered = list(compress(dataset, mask))
    matrices_filt_dir = mat_dir[mask_tensor]
    matrices_filt_rc = mat_rc[mask_tensor]
    
    scored_regions = []
    valid_indices = list(range(1, 34))

    for is_rc, tensor in [(False, matrices_filt_dir), (True, matrices_filt_rc)]:
        for i, sample in enumerate(data_filtered):
            seq = str(Seq(sample['sequence']).reverse_complement()) if is_rc else sample['sequence']
            attn_row = tensor[i][0, :]
            
            for attn_idx in valid_indices:
                score = attn_row[attn_idx].item()
                token_start = (attn_idx - 1) * 6
                start_ctx = max(0, token_start - context_flank)
                end_ctx = min(len(seq), token_start + 6 + context_flank)
                region_seq = seq[start_ctx:end_ctx]
                
                if len(region_seq) >= 12:
                    scored_regions.append((score, region_seq))
                    
    scored_regions.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored_regions[:top_k]]

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================
# (I plot Kmers e Positions rimangono quelli puliti del messaggio precedente)
def plot_top_kmers(dataset, avg_dir, avg_rc, args):
    print("[*] Generazione K-mer Lollipop Plot...")
    valid_indices = list(range(1, 34))
    token_attn, token_counts = defaultdict(float), defaultdict(int)

    for is_rc, att_tensor in [(False, avg_dir), (True, avg_rc)]:
        for i, sample in enumerate(dataset):
            seq = str(Seq(sample['sequence']).reverse_complement()) if is_rc else sample['sequence']
            tokens = tokenize_dna_sequence(seq)
            attn_received = att_tensor[i][0, :]
            
            for pos in valid_indices:
                if pos >= len(tokens): break
                t = tokens[pos]
                token_attn[t] += attn_received[pos].item()
                token_counts[t] += 1

    mean_attn = {k: token_attn[k] / token_counts[k] for k in token_attn}
    sym_attn = {}
    used = set()
    for k, v in mean_attn.items():
        if k in used: continue
        k_rc = str(Seq(k).reverse_complement())
        if k_rc in mean_attn:
            score = mean_attn[k] + mean_attn[k_rc] if k != k_rc else mean_attn[k]
            key = '\n'.join(sorted([k, k_rc])) if k != k_rc else k
            used.update([k, k_rc])
        else:
            score, key = v, k
            used.add(k)
        sym_attn[key] = score

    sorted_items = sorted(sym_attn.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
    labels = [k.replace('\n', ' / ') for k, _ in sorted_items][::-1]
    scores = [v for _, v in sorted_items][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = mcolors.LinearSegmentedColormap.from_list("TealToOrange", ['#006B7D', '#D4880F'], N=256)
    colors = cmap(norm(scores))
    y_pos = range(len(labels))

    ax.hlines(y=y_pos, xmin=0, xmax=scores, color=colors, alpha=0.4, linewidth=2)
    ax.scatter(scores, y_pos, color=colors, s=150, alpha=1.0, edgecolor='white', linewidth=1.5)
    ax.axvline(np.mean(scores), color='#555555', linestyle='--', alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, family='monospace')
    ax.set_xlabel("Mean Attention Weight", fontweight='bold')
    ax.set_title(f"Top {args.top_k} Most Attended 6-mers ({args.cell_type.upper()})", fontweight='bold')
    
    for i, (score, y) in enumerate(zip(scores, y_pos)):
        ax.text(score + (max(scores)*0.02), y, f'{score:.3f}', va='center', fontsize=9)
    # sns.set_style({"axes.facecolor": '#FDF7F7'})
    # fig.set_facecolor('#FDF7F7')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'top{args.top_k}_kmers_{args.cell_type}_{args.filter}_MASK.png'))
    plt.close()
    print("[+] Plot salvato.")

def plot_position_importance(avg_dir, avg_rc, args):
    print("[*] Generazione Position Importance Bar Plot...")
    attn_dir = avg_dir[:, 0, 1:34]
    attn_rc = torch.flip(avg_rc[:, 0, 1:34], dims=[1])
    attn_tot = torch.cat([attn_dir, attn_rc], axis=0)

    mean_attn = torch.mean(attn_tot, dim=0).cpu().numpy()
    sem_attn = torch.std(attn_tot, dim=0).cpu().numpy() / np.sqrt(attn_tot.shape[0])
    positions = np.arange(1, 34)
    labels = [f'{(p-1)*6}–{p*6}' for p in positions]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(positions, mean_attn, color='#006B7D', edgecolor='#004D5C', alpha=0.85)
    ax.errorbar(positions, mean_attn, yerr=sem_attn, fmt='none', ecolor='#333333', alpha=0.6)

    thresh = np.percentile(mean_attn, 75)
    for bar, val in zip(bars, mean_attn):
        if val >= thresh: bar.set_color('#D4880F')

    ax.set_xticks(positions[::2])
    ax.set_xticklabels(labels[::2], rotation=45, ha="right")
    ax.set_xlabel("Sequence Position (bp)", fontweight='semibold')
    ax.set_ylabel("Mean Attention Weight", fontweight='semibold')
    plt.tight_layout()
    # fig.set_facecolor('#FDF7F7')
    # sns.set_style({"axes.facecolor": '#FDF7F7'})

    plt.savefig(os.path.join(args.out_dir, f'position_importance_{args.cell_type}_MASK.png'))
    plt.close()
    print("[+] Plot salvato.")


def plot_position_importance_symmetric(avg_dir, avg_rc, args):
    print("[*] Generazione Position Importance Bar Plot Simmetrico (Attenzione)...")
    
    # 1. Estrazione e allineamento (come facevi già)
    attn_dir = avg_dir[:, 0, 1:34]
    attn_rc = torch.flip(avg_rc[:, 0, 1:34], dims=[1])
    attn_tot = torch.cat([attn_dir, attn_rc], axis=0)

    # 2. Calcolo media e deviazione standard originali
    mean_attn = torch.mean(attn_tot, dim=0).cpu().numpy()
    sem_attn = torch.std(attn_tot, dim=0).cpu().numpy() / np.sqrt(attn_tot.shape[0])

    # =========================================================================
    # 3. SIMMETRIZZAZIONE STRUTTURALE (Come fatto in IG)
    # =========================================================================
    mean_attn_sym = (mean_attn + mean_attn[::-1]) / 2.0
    sem_attn_sym = (sem_attn + sem_attn[::-1]) / 2.0

    # 4. Preparazione per il plot
    positions = np.arange(1, 34)
    labels = [f'{(p-1)*6}–{p*6}' for p in positions]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Usa le variabili simmetrizzate per il plot
    bars = ax.bar(positions, mean_attn_sym, color='#006B7D', edgecolor='#004D5C', alpha=0.85)
    ax.errorbar(positions, mean_attn_sym, yerr=sem_attn_sym, fmt='none', ecolor='#333333', alpha=0.6)

    # Threshold per colorare i picchi (basato sulla distribuzione simmetrica)
    thresh = np.percentile(mean_attn_sym, 75)
    for bar, val in zip(bars, mean_attn_sym):
        if val >= thresh: 
            bar.set_color('#D4880F')

    # Formatting dell'asse X
    ax.set_xticks(positions[::2])
    ax.set_xticklabels(labels[::2], rotation=45, ha="right")
    ax.set_xlabel("Sequence Position (bp)", fontweight='semibold')
    ax.set_ylabel("Mean Attention Weight", fontweight='semibold')
    # fig.set_facecolor('#FDF7F7')
    # sns.set_style({"axes.facecolor": '#FDF7F7'})    
    plt.tight_layout()
    save_path = os.path.join(args.out_dir, f'position_importance_symmetric_{args.cell_type}_MASK.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[+] Plot salvato in: {save_path}")

def plot_biophysical_profiles(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args):
    print(f"[*] Generazione Biophysical Profiles ({args.region.upper()} region)...")
    # mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    # mask_tn = [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)]

    mask_pp = [p == 1 for p in preds_int]
    mask_pn = [p == 0 for p in preds_int]
    
    # seq_pos = extract_top_kmers(mask_tp, dataset_all, avg_dir_all, avg_rc_all, args, top_k=100)
    # seq_neg = extract_top_kmers(mask_tn, dataset_all, avg_dir_all, avg_rc_all, args, top_k=100)
    seq_pos = extract_top_kmers(mask_pp, dataset_all, avg_dir_all, avg_rc_all, args, top_k=100)
    seq_neg = extract_top_kmers(mask_pn, dataset_all, avg_dir_all, avg_rc_all, args, top_k=100)
    
    all_seqs = [d['sequence'] for d in dataset_all]
    seq_sample = random.sample(all_seqs, min(2000, len(all_seqs)))
    raw_bg = [s[i:i+6] for s in seq_sample for i in range(len(s)-5)]
    seq_bg = random.sample(raw_bg, min(50000, len(raw_bg)))

    data_plot = []
    for seq, group in zip([seq_pos, seq_bg, seq_neg], ['Nucleosome Anchors\n(Predicted Positives)', 'Global Background', 'Nucleosome Exclusion\n(Predicted Negatives)']):
        for s in seq:
            res = calculate_physics_complete(s)
            res['Group'] = group
            data_plot.append(res)

    df_compare = pd.DataFrame(data_plot)
    order_plot = ['Nucleosome Anchors\n(Predicted Positives)', 'Global Background', 'Nucleosome Exclusion\n(Predicted Negatives)']
    palette = {'Nucleosome Anchors\n(Predicted Positives)': '#E08E00', 'Global Background': '#A9A9A9', 'Nucleosome Exclusion\n(Predicted Negatives)':'#008C9E'}
    
    metrics = ['Propeller Twist', 'Static Bending', 'DNA Roll', 'Minor Groove Width', 'Helical Twist', 'Major Groove Width']
    ylabels = ["Degrees\n(Less Neg = Flexible)", "Degrees\n(High = Kink)", "Degrees\n(High = Open/Curved)", "Angstroms (Å)\n(Low = Narrow)", "Degrees\n(High = Tighter Helix)", "Angstroms (Å)\n(High = Wide)"]

    fig, axes = plt.subplots(3, 2, figsize=(16,21))
    for i, metric in enumerate(metrics):
        ax = axes.flatten()[i]
        sns.violinplot(data=df_compare, x='Group', y=metric, order=order_plot, palette=palette, ax=ax, inner="quartile", cut=0)
        
        vals_pos = df_compare[df_compare['Group']=='Nucleosome Anchors\n(Predicted Positives)'][metric]
        vals_bg = df_compare[df_compare['Group']=='Global Background'][metric]
        vals_neg = df_compare[df_compare['Group']=='Nucleosome Exclusion\n(Predicted Negatives)'][metric]
        
        _, p_pos = mannwhitneyu(vals_pos, vals_bg, alternative='two-sided')
        _, p_neg = mannwhitneyu(vals_neg, vals_bg, alternative='two-sided')
        
        y_max, y_range = df_compare[metric].max(), df_compare[metric].max() - df_compare[metric].min()
        h = y_range * 0.03
        
        def annot_stat(x1, x2, y, h, p):
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
            ax.text((x1+x2)*.5, y+h, "p < 0.001" if p < 0.001 else f"p = {p:.2e}", ha='center', va='bottom', color='k', fontsize=15, fontweight='bold')

        annot_stat(0, 1, y_max + (y_range * 0.05), h, p_pos)
        annot_stat(1, 2, y_max + (y_range * 0.17), h, p_neg)

        ax.set_title(metric, fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel(ylabels[i], fontsize=15)
        ax.set_xlabel("")
        ax.set_ylim(top=y_max + (y_range * 0.35))
        ax.set_xticklabels(['Nucleosome Anchoring\n(PP)', 'Global\nBackground', 'Nucleosome Exclusion\n(PN)'], fontsize=15)
        ax.tick_params(axis='y', labelsize=14)

    if args.cell_type.upper() == 'ACT':
        plt.suptitle(f"Biophysical Profiles in Activated Cells: Predicted Positives vs Predicted Negatives {args.region.capitalize()} Region Analysis", fontsize=21, y=1.04, fontweight='bold')
    elif args.cell_type.upper() == 'REST':
        plt.suptitle(f"Biophysical Profiles in Resting Cells: Predicted Positives vs Predicted Negatives {args.region.capitalize()} Region Analysis", fontsize=21, y=1.04, fontweight='bold')  
    else:
        plt.suptitle(f"Biophysical Profiles in Limphoblastoid Cells: Predicted Positives vs Predicted Negatives {args.region.capitalize()} Region Analysis", fontsize=21, y=1.04, fontweight='bold')          

    # fig.set_facecolor('#FDF7F7')
    # sns.set_style({"axes.facecolor": '#FDF7F7'})
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)
    plt.savefig(os.path.join(args.out_dir, f'biophysical_profiles_{args.region}_{args.cell_type}_MASK.png'), bbox_inches='tight')
    plt.close()
    print("[+] Plot salvato.")

def plot_motif_enrichment(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args):
    print("[*] Download JASPAR e scansione Motif Enrichment (potrebbe richiedere tempo)...")
    # mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    # mask_tn = [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)]

    mask_pp = [p == 1 for p in preds_int]
    mask_pn = [p == 0 for p in preds_int]
    
    # seqs_pos = extract_attended_regions(mask_tp, dataset_all, avg_dir_all, avg_rc_all, top_k=100)
    # seqs_neg = extract_attended_regions(mask_tn, dataset_all, avg_dir_all, avg_rc_all, top_k=100)

    seqs_pos = extract_attended_regions(mask_pp, dataset_all, avg_dir_all, avg_rc_all, top_k=100)
    seqs_neg = extract_attended_regions(mask_pn, dataset_all, avg_dir_all, avg_rc_all, top_k=100)
    
    full_text = "".join(seqs_pos + seqs_neg)
    tot = len(full_text)
    bg_dist = {n: full_text.count(n)/tot for n in 'ACGT'}
    
    jaspar_file = "JASPAR2022_CORE_vertebrates.txt"
    if not os.path.exists(jaspar_file) or os.path.getsize(jaspar_file) < 1000:
        url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
        with open(jaspar_file, 'wb') as f: f.write(requests.get(url).content)
            
    with open(jaspar_file) as handle:
        motif_list = [m for m in motifs.parse(handle, "jaspar") if m.counts is not None and len(m.counts) > 0]

    results, total_pos, total_neg = [], len(seqs_pos), len(seqs_neg)
    
    for m in tqdm(motif_list, desc="Scansione Motivi"):
        try:
            m.background = bg_dist
            pssm = m.counts.normalize(pseudocounts=0.5).log_odds(bg_dist)
            threshold = pssm.max * 0.80
            
            hits_pos = sum(1 for seq in seqs_pos if len(seq) >= m.length and max(pssm.calculate(Seq(seq))) > threshold)
            hits_neg = sum(1 for seq in seqs_neg if len(seq) >= m.length and max(pssm.calculate(Seq(seq))) > threshold)
            
            if (hits_pos + hits_neg) < 3: continue
            
            _, p_val = fisher_exact([[hits_pos, total_pos - hits_pos], [hits_neg, total_neg - hits_neg]])
            freq_pos, freq_neg = hits_pos / total_pos, hits_neg / total_neg
            log2fc = np.log2((freq_pos + 1e-6) / (freq_neg + 1e-6))
            
            results.append({'ID': m.matrix_id, 'Name': m.name, 'Log2FC': log2fc, 'P-value': p_val})
        except: continue

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[-] Nessun motivo significativo trovato.")
        return

    _, df_res['FDR'], _, _ = multipletests(df_res['P-value'], method='fdr_bh')
    df_res['-log10 FDR'] = -np.log10(df_res['FDR'] + 1e-100)
    
    plt.figure(figsize=(8, 5))
    colors = df_res.apply(lambda r: '#D4880F' if r['Log2FC']>0.5 and r['FDR']<=0.05 else ('#006B7D' if r['Log2FC']<-0.5 and r['FDR']<=0.05 else '#E0E0E0'), axis=1)
    sns.scatterplot(data=df_res, x='Log2FC', y='-log10 FDR', c=colors, s=100, alpha=0.9, edgecolor='black')
    
    plt.axhline(-np.log10(0.05), linestyle='--', color='gray')
    plt.axvline(0.5, linestyle=':', color='gray'); plt.axvline(-0.5, linestyle=':', color='gray')
    
    top_hits = pd.concat([df_res[(df_res['Log2FC']>0.5) & (df_res['FDR']<=0.05)].head(2), df_res[(df_res['Log2FC']<-0.5) & (df_res['FDR']<=0.05)].head(2)])
    for _, row in top_hits.iterrows():
        plt.text(row['Log2FC'], row['-log10 FDR']+0.6, row['Name'], fontsize=9, fontweight='bold', ha='center')
    
    plt.title(f"Motif Enrichment in Top Attended Regions ({args.cell_type.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("Log2 Fold Change (Linker <-> Nucleosome)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'motif_enrichment_{args.cell_type}:MASK.png'), bbox_inches='tight')
    plt.close()
    print("[+] Plot salvato.")

def plot_biophysical_metaprofiles(dataset_all, preds_int, val_labels, args):
    print("[*] Generazione Biophysical Metaprofiles (Line plots)...")
    
    # 1. Filtriamo TP e TN
    mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    mask_tn = [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)]
    
    seqs_tp = [d['sequence'] for d in compress(dataset_all, mask_tp)]
    seqs_tn = [d['sequence'] for d in compress(dataset_all, mask_tn)]
    
    # Dizionario delle metriche
    metrics = {
        'Propeller Twist': PROP_TWIST_DIPRO,
        'Static Bending': BENDABILITY,
        'DNA Roll': ROLL_DIPRO,
        'Minor Groove Width': MGW,
        'Helical Twist': HELICAL_TWIST,
        'Major Groove Width': MAJOR_GW
    }
    
    ylabels = [
        "Degrees (Less Neg = Flexible)", "Degrees (High = Kink)", 
        "Degrees (High = Open)", "Angstroms (Å) (Low = Narrow)", 
        "Degrees (High = Tighter Helix)", "Angstroms (Å) (High = Wide)"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Finestra di smoothing (5 bp è l'ideale per evidenziare l'onda da 10.5 bp)
    window_size = 5 
    
    for i, (metric_name, metric_dict) in enumerate(metrics.items()):
        ax = axes.flatten()[i]
        
        # Estrazione profili per tutti i campioni
        profiles_tp = np.array([get_sequence_shape_profile(s, metric_dict) for s in seqs_tp])
        profiles_tn = np.array([get_sequence_shape_profile(s, metric_dict) for s in seqs_tn])
        
        # Calcolo della media per posizione (ignora i NaN derivati dalle 'N' se presenti)
        mean_tp = np.nanmean(profiles_tp, axis=0)
        mean_tn = np.nanmean(profiles_tn, axis=0)
        
        # Smoothing per pulire il grafico
        smooth_tp = smooth_profile(mean_tp, window=window_size)
        smooth_tn = smooth_profile(mean_tn, window=window_size)
        
        # Asse X centrato sulla diade
        # Se la sequenza è di 201bp (200 step), la diade è a step 100.
        # Con valid smoothing, l'array si accorcia, quindi aggiustiamo l'asse X
        offset = window_size // 2
        x_axis = np.arange(len(smooth_tp)) - (100 - offset)
        
        # Plot
        ax.plot(x_axis, smooth_tp, label='Nucleosome Anchoring (TP)', color='#E08E00', linewidth=2.5)
        ax.plot(x_axis, smooth_tn, label='Nucleosome Exclusion (TN)', color='#008C9E', linewidth=2.5, linestyle='--')
        
        # Evidenziamo il centro (Diade)
        ax.axvline(0, color='#666666', linestyle=':', alpha=0.5)
        
        # Styling
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabels[i], fontsize=11)
        ax.set_xlabel("Distance from Dyad (bp)", fontsize=11)
        if i == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=11)

    plt.suptitle(f"Biophysical Metaprofiles in {args.cell_type.upper()}: Periodic Wave Signatures", 
                 fontsize=18, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'biophysical_metaprofiles_{args.cell_type}_MASK.png'), bbox_inches='tight')
    plt.close()
    print("[+] Plot Metaprofilo salvato.")

def plot_attention_driven_physics(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args):
    print("[*] Generazione Attention-Driven Physics Profiles (High vs Low Attn in TP)...")
    
    # Filtriamo SOLO i True Positives
    mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    mask_tensor = torch.tensor(mask_tp, dtype=torch.bool)
    
    data_tp = list(compress(dataset_all, mask_tp))
    attn_dir_tp = avg_dir_all[mask_tensor]
    attn_rc_tp = avg_rc_all[mask_tensor]
    
    # Indici validi del CORE (escludiamo le N e i primissimi/ultimissimi k-mer)
    # Le N erano [18:36] (k-mer 4,5,6) e [162:180] (k-mer 28,29,30)
    # Prendiamo solo il nucleosoma centrale (k-mer dal 7 al 27)
    core_indices = list(range(7, 28))
    
    high_attn_physics = []
    low_attn_physics = []
    
    for i, sample in enumerate(data_tp):
        seq = sample['sequence']
        
        # Simmetrizziamo e combiniamo l'attenzione per il campione 'i'
        a_dir = attn_dir_tp[i][0, :].cpu().numpy()
        a_rc = torch.flip(attn_rc_tp[i][0, :], dims=[0]).cpu().numpy()
        a_tot = (a_dir + a_rc) / 2.0
        
        # Estraiamo l'attenzione solo per il core
        core_attn = [(idx, a_tot[idx]) for idx in core_indices]
        
        # Ordiniamo gli indici in base all'attenzione
        core_attn.sort(key=lambda x: x[1], reverse=True)
        
        # Prendiamo i Top 3 e Bottom 3
        top_indices = [x[0] for x in core_attn[:3]]
        bottom_indices = [x[0] for x in core_attn[-3:]]
        
        # Estraiamo le sequenze (6-mer) corrispondenti
        for idx in top_indices:
            start = (idx - 1) * 6
            kmer = seq[start:start+6]
            if 'N' not in kmer:  # Controllo di sicurezza
                res = calculate_physics_complete(kmer)
                res['Attention Level'] = 'High Attention\n(Top 3 k-mers)'
                high_attn_physics.append(res)
                
        for idx in bottom_indices:
            start = (idx - 1) * 6
            kmer = seq[start:start+6]
            if 'N' not in kmer:
                res = calculate_physics_complete(kmer)
                res['Attention Level'] = 'Ignored\n(Bottom 3 k-mers)'
                low_attn_physics.append(res)

    # Creazione DataFrame e Plot
    df_plot = pd.DataFrame(high_attn_physics + low_attn_physics)
    metrics = ['Propeller Twist', 'Static Bending', 'DNA Roll', 'Minor Groove Width', 'Helical Twist', 'Major Groove Width']
    ylabels = ["Degrees\n(Less Neg = Flexible)", "Degrees\n(High = Kink)", "Degrees\n(High = Open/Curved)", "Angstroms (Å)\n(Low = Narrow)", "Degrees\n(High = Tighter Helix)", "Angstroms (Å)\n(High = Wide)"]
    palette = {'High Attention\n(Top 3 k-mers)': '#D4880F', 'Ignored\n(Bottom 3 k-mers)': '#555555'}

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    for j, metric in enumerate(metrics):
        ax = axes.flatten()[j]
        sns.violinplot(data=df_plot, x='Attention Level', y=metric, palette=palette, ax=ax, inner="quartile", cut=0)
        
        # Statistica
        vals_high = df_plot[df_plot['Attention Level']=='High Attention\n(Top 3 k-mers)'][metric]
        vals_low = df_plot[df_plot['Attention Level']=='Ignored\n(Bottom 3 k-mers)'][metric]
        _, p_val = mannwhitneyu(vals_high, vals_low, alternative='two-sided')
        
        y_max = df_plot[metric].max()
        y_range = df_plot[metric].max() - df_plot[metric].min()
        
        # Annotazione p-value
        ax.plot([0, 0, 1, 1], [y_max + y_range*0.05, y_max + y_range*0.08, y_max + y_range*0.08, y_max + y_range*0.05], lw=1.5, c='k')
        ax.text(0.5, y_max + y_range*0.09, "p < 0.001" if p_val < 0.001 else f"p = {p_val:.2e}", ha='center', va='bottom', color='k', fontsize=15, fontweight='bold')
        
        ax.set_title(metric, fontsize=15, fontweight='bold')
        ax.set_ylabel(ylabels[j], fontsize=15)
        ax.set_xlabel("")
        ax.set_ylim(top=y_max + y_range*0.2)

    plt.suptitle(f"What the Model is Actually Looking At ({args.cell_type.upper()})\nHigh vs Low Attended Regions within True Positives", fontsize=18, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'attention_driven_physics_{args.cell_type}_MASK.png'), bbox_inches='tight')
    plt.close()
    print("[+] Plot salvato.")



def plot_nucleotide_composition(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args):
    print(f"[*] Generazione Nucleotide Composition Profiling ({args.region.upper()} region)...")
    
    # Maschere per TP e TN
    # mask_tp = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)]
    # mask_tn = [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)]

    mask_pp = [p == 1 for p in preds_int]
    mask_pn = [p == 0 for p in preds_int]
    
    # Estraiamo i top k-mers (usando la tua funzione esistente)
    # seq_pos = extract_top_kmers(mask_tp, dataset_all, avg_dir_all, avg_rc_all, args, top_k=300)
    # seq_neg = extract_top_kmers(mask_tn, dataset_all, avg_dir_all, avg_rc_all, args, top_k=300)

    seq_pos = extract_top_kmers(mask_pp, dataset_all, avg_dir_all, avg_rc_all, args, top_k=300)
    seq_neg = extract_top_kmers(mask_pn, dataset_all, avg_dir_all, avg_rc_all, args, top_k=300)
    
    # Creiamo un background globale di riferimento
    all_seqs = [d['sequence'] for d in dataset_all]
    seq_sample = random.sample(all_seqs, min(2000, len(all_seqs)))
    raw_bg = [s[i:i+6] for s in seq_sample for i in range(len(s)-5)]
    # Escludiamo le N dal background
    seq_bg = random.sample([s for s in raw_bg if 'N' not in s], min(10000, len(raw_bg)))

    # Sotto-funzione per contare le frequenze dei dinucleotidi
    def get_dinucleotide_freqs(seq_list):
        di_counts = {n1+n2: 0 for n1 in 'ACGT' for n2 in 'ACGT'}
        tot_di = 0
        for seq in seq_list:
            if 'N' in seq: continue # Ignoriamo eventuali frammenti con mascheramento
            for i in range(len(seq)-1):
                di = seq[i:i+2]
                if di in di_counts:
                    di_counts[di] += 1
                    tot_di += 1
        return {k: (v / tot_di) * 100 if tot_di > 0 else 0 for k, v in di_counts.items()}

    # Calcoliamo le frequenze percentuali
    di_pos = get_dinucleotide_freqs(seq_pos)
    di_neg = get_dinucleotide_freqs(seq_neg)
    di_bg = get_dinucleotide_freqs(seq_bg)

    # Prepariamo i dati per Seaborn
    data = []
    for k in di_pos.keys():
        data.append({'Dinucleotide': k, 'Frequency (%)': di_pos[k], 'Group': 'Nucleosome Anchoring\n(Predicted Positives)'})
        data.append({'Dinucleotide': k, 'Frequency (%)': di_bg[k], 'Group': 'Global Background'})
        data.append({'Dinucleotide': k, 'Frequency (%)': di_neg[k], 'Group': 'Nucleosome Exclusion\n(Predicted Negatives)'})

    df_comp = pd.DataFrame(data)

    # Ordiniamo i dinucleotidi per la differenza tra TN e TP (dal più tipico dei TN al più tipico dei TP)
    diff_dict = {k: di_neg[k] - di_pos[k] for k in di_pos.keys()}
    sorted_dinucl = sorted(diff_dict.keys(), key=lambda x: diff_dict[x], reverse=True)

    # Plot
    fig=plt.figure(figsize=(14, 6))
    palette = {'Nucleosome Anchoring\n(Predicted Positives)': '#E08E00', 
               'Global Background': '#A9A9A9', 
               'Nucleosome Exclusion\n(Predicted Negatives)':'#008C9E'}
    
    sns.barplot(data=df_comp, x='Dinucleotide', y='Frequency (%)', hue='Group', 
                order=sorted_dinucl, palette=palette, edgecolor='black', alpha=0.9)
    
    plt.title(f"Dinucleotide Content of the Most Attended Regions ({args.cell_type.upper()})", 
              fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Relative Frequency (%)", fontsize=15, fontweight='bold')
    plt.xlabel("Dinucleotide Step", fontsize=15, fontweight='bold')
    
    plt.legend(title="", fontsize=15, frameon=False)
    sns.despine()
    # fig.set_facecolor('#FDF7F7')
    # sns.set_style({"axes.facecolor": '#FDF7F7'})
    
    plt.tight_layout()
    save_path = os.path.join(args.out_dir, f'nucleotide_composition_{args.region}_{args.cell_type}_MASK.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[+] Plot Composizione Nucleotidica salvato in: {save_path}")










def applica_mascheramento(dataset):
    """Sostituisce con 'N' gli indici [18:36] e [162:180]"""
    for d in dataset:
        seq = d['sequence']
        masked_seq = seq[:18] + ('N' * 18) + seq[36:162] + ('N' * 18) + seq[180:]
        d['sequence'] = masked_seq
    return dataset

    
# ==============================================================================
# 5. MAIN LOGIC E CARICAMENTO
# ==============================================================================
def load_data_and_matrices(args):
    print(f"[*] Caricamento dataset: {args.cell_type.upper()}")
    ds_map = {
        'lympho': '../../INFERENCE/NuPoSe_Yosef_Data/Lympho_Nucleosomal_TOT.pkl', 
        'act': '../../INFERENCE/NuPoSe_Yosef_Data/ActivatingNuclesomal_TOT.pkl', 
        'rest': '../../INFERENCE/NuPoSe_Yosef_Data/RestingNuclesomal_TOT.pkl' # Corretto resting maiuscolo/minuscolo se serve
    }
    
    # Nomi esatti per le predizioni
    pred_map = {
        'lympho': 'preds_lymphoblastoid_best_model_YOSEF.pkl', 
        'act': 'preds_act_best_model_YOSEF.pkl', 
        'rest': 'preds_rest_best_model_YOSEF.pkl' # <-- Inserisci il nome corretto qui se manca
    }
    
    # Mappatura per il nome del file .pt
    full_name_map = {
        'lympho': 'lymphoblastoid', 
        'act': 'act', 
        'rest': 'rest'
    }

    # Caricamento Dataset
    with open(os.path.join(args.data_dir, ds_map[args.cell_type]), 'rb') as f: 
        dataset = pickle.load(f)

    # dataset = applica_mascheramento(dataset)  #MASKERA DISATTIVATA !
    
    for d in dataset: 
        d['sequence_rev'] = str(Seq(d['sequence']).reverse_complement())

    # Caricamento Matrici (Ora è DINAMICO!)
    full_name = full_name_map[args.cell_type]
    file_pt = os.path.join(args.results_dir, f"matrices_results_{full_name}_best_model_YOSEF.pt")
    
    print(f"[*] Caricamento matrici da: {file_pt}")
    data = torch.load(file_pt, map_location='cpu')
    
    total_matrices_dir = torch.stack(data['matrices_dir'])
    total_matrices_rc = torch.stack(data['matrices_rc'])
            
    avg_dir, avg_rc = total_matrices_dir, total_matrices_rc

    # Caricamento Predizioni
    file_preds = os.path.join(args.results_dir, pred_map[args.cell_type])
    preds_int = (np.array(pd.read_pickle(file_preds)) > 0.5).astype(int)
    
    return dataset, avg_dir, avg_rc, preds_int

def main():
    parser = argparse.ArgumentParser(description="Explainability Tool")
    parser.add_argument("--data_dir", type=str, default="../data/data_pkl", help="Cartella dei .pkl")
    parser.add_argument("--results_dir", type=str, default="../results", help="Cartella dei .pt e predizioni")
    parser.add_argument("--out_dir", type=str, default="../images", help="Cartella per salvare le immagini")
    parser.add_argument("--cell_type", type=str, choices=['lympho', 'act', 'rest'], required=True)
    parser.add_argument("--filter", type=str, choices=['all', 'tp', 'tn'], default='all')
    parser.add_argument("--region", type=str, choices=['dyad', 'boundary', 'global', 'shoulder'], default='global')
    parser.add_argument("--top_k", type=int, default=20)
    
    parser.add_argument("--all_plots", action="store_true", help="Esegui tutti i plot")
    parser.add_argument("--plot_kmers", action="store_true")
    parser.add_argument("--plot_positions", action="store_true")
    parser.add_argument("--plot_physics", action="store_true", help="Violin plots delle proprietà biofisiche")
    parser.add_argument("--plot_motifs", action="store_true", help="Volcano plot per enrichment JASPAR")
    parser.add_argument("--plot_composition", action="store_true", help="Bar plot della composizione nucleotidica (dinucleotidi)")
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)


    if args.all_plots:
        args.plot_kmers = args.plot_positions = args.plot_physics = args.plot_motifs = args.plot_composition = True


    
    dataset_all, avg_dir_all, avg_rc_all, preds_int = load_data_and_matrices(args)
    val_labels = [s['label'] for s in dataset_all]

    # Masking per K-mers e Positions
    mask = [(p == 1 and tl == 1) for p, tl in zip(preds_int, val_labels)] if args.filter == 'tp' else \
           [(p == 0 and tl == 0) for p, tl in zip(preds_int, val_labels)] if args.filter == 'tn' else [True] * len(val_labels)
    
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    dataset_filtered = list(compress(dataset_all, mask))
    avg_dir_filtered, avg_rc_filtered = avg_dir_all[mask_tensor], avg_rc_all[mask_tensor]

    if args.plot_kmers: plot_top_kmers(dataset_filtered, avg_dir_filtered, avg_rc_filtered, args)
    if args.plot_positions: plot_position_importance_symmetric(avg_dir_filtered, avg_rc_filtered, args)   #HO USATO SIMMETRIA!!!
    
    # Physics e Motifs richiedono l'intero dataset per calcolare il background e fare confronti
    if args.plot_physics: plot_biophysical_profiles(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args)
    if args.plot_motifs: plot_motif_enrichment(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args)
    if args.plot_composition: plot_nucleotide_composition(dataset_all, avg_dir_all, avg_rc_all, preds_int, val_labels, args)
if __name__ == "__main__":
    main()