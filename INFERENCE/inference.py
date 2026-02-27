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

import torch
from torch.utils.data import DataLoader
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

def output_model_batch_inference(batch, model, tokenizer, device, rc=True):
    sequences_fw = list(batch['sequence'])
    
    tokens_fw = tokenizer(
        sequences_fw, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    input_ids = tokens_fw['input_ids'].to(device)
    attention_mask = tokens_fw['attention_mask'].to(device)
    
    output, importance = model(input_ids, attention_mask)

    if rc:
        if 'sequence_rev' in batch:
            sequences_rc = list(batch['sequence_rev'])
        else:
            sequences_rc = sequences_fw
            
        tokens_rc = tokenizer(
            sequences_rc, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        input_ids_rc = tokens_rc['input_ids'].to(device)
        attention_mask_rc = tokens_rc['attention_mask'].to(device)
        
        output_rc, importance_rc = model(input_ids_rc, attention_mask_rc)
    else:
        output_rc, importance_rc = output, importance

    return output, output_rc, importance, importance_rc

def predict_only(model, dataloader, tokenizer, device, threshold=0.5, return_importance=True):
    model.eval()
    all_probs = []
    all_importance = []
    
    print(f"[*] Avvio inferenza su device: {device} (Tokenizzazione on-the-fly)")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            output, output_rc, imp, _ = output_model_batch_inference(
                batch, model, tokenizer, device, rc=True
            )
            
            probs = torch.sigmoid((output + output_rc) / 2)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            
            if return_importance and imp is not None:
                all_importance.append(imp.cpu().numpy())

            del output, output_rc, probs
            if imp is not None: del imp

    probs_np = np.array(all_probs)
    preds_np = (probs_np >= threshold).astype(int)
    
    result = {
        'probabilities': probs_np,
        'predictions': preds_np
    }
    
    if return_importance:
        result['importance'] = all_importance
        
    return result

def plot_attention_only(sample_idx, dataset, matrices, output_dir, k=6):
    sample = dataset[sample_idx]
    seq = sample['sequence']
    attn_data = matrices[sample_idx]
    
    if hasattr(attn_data, 'numpy'):
        attn_data = attn_data.numpy()
        
    if attn_data.ndim > 1:
        attn_values = attn_data[0, 1:-3]
    else:
        attn_values = attn_data[1:-3] 
    
    tokens = [seq[i:i+k] for i in range(0, k * len(attn_values), k)]
    min_len = min(len(tokens), len(attn_values))
    tokens = tokens[:min_len]
    attn_values = attn_values[:min_len]
    
    df_plot = pd.DataFrame({
        'Index': range(len(tokens)),
        'Token': tokens,
        'Attention': attn_values
    })

    plt.figure(figsize=(20, 8))
    colors = ['#006B7D'] * len(tokens)
    ax = sns.barplot(data=df_plot, x='Token', y='Attention', palette=colors)
    
    ax.set_ylabel('Attention Score', fontsize=18, fontweight='bold')
    ax.set_xlabel('Sequence Tokens (6-mers)', fontsize=18, fontweight='bold')
    plt.xticks(rotation=90, ha='center', fontfamily='monospace', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Salva invece di mostrare
    save_path = os.path.join(output_dir, f"attention_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Plot attenzione salvato in: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for CadmusDNA")
    parser.add_argument("--dataset", type=str, required=True, help="Path al file del dataset (pickle)")
    parser.add_argument("--weights", type=str, required=True, help="Path al file dei pesi del modello (.pt)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Cartella di destinazione per i risultati")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch per l'inferenza")
    
    args = parser.parse_args()

    # Creazione directory di output
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Caricamento Dati
    print(f"[*] Caricamento dataset da: {args.dataset}")
    dataset = pd.read_pickle(args.dataset)
    
    # Aggiungo la reverse complement se mancante
    for d_entry in dataset:
        if 'sequence_rev' not in d_entry:
            d_entry['sequence_rev'] = str(Seq(d_entry['sequence']).reverse_complement())
            
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Inizializzazione Modello
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

    # 3. Caricamento Tokenizer
    model_name = "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 4. Inferenza
    results = predict_only(model, test_loader, tokenizer, device)

    # 5. Salvataggio Risultati
    print("[*] Salvataggio dei risultati in corso...")
    
    # Salvataggio Probabilità
    df_results = pd.DataFrame({
        'probabilities': results['probabilities'],
        'predictions': results['predictions']
    })
    
    # Se le label sono presenti nel dataset originale, le aggiungiamo
    if 'label' in dataset[0]:
        df_results['true_label'] = [d['label'] for d in dataset]
        
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[*] Predizioni salvate in: {csv_path}")

    # Processamento Matrici di Attenzione
    if 'importance' in results:
        flat_importance = []
        for batch_matrix in results['importance']:
            for sample_matrix in batch_matrix:
                flat_importance.append(sample_matrix)
                
        # Salva le matrici grezze in un pickle per future analisi
        matrices_path = os.path.join(args.output_dir, "attention_matrices.pkl")
        with open(matrices_path, 'wb') as f:
            pickle.dump(flat_importance, f)
        print(f"[*] Matrici di attenzione salvate in: {matrices_path}")

        # Plotta l'attenzione per il primo campione come esempio
        if len(dataset) > 0:
            plot_attention_only(
                sample_idx=0, 
                dataset=dataset, 
                matrices=flat_importance, 
                output_dir=args.output_dir
            )

    print("[*] Processo completato con successo!")

if __name__ == "__main__":
    main()