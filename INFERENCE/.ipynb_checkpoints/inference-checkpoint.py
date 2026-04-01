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

def map_attention_to_bp(imp_array, input_ids_tensor, tokenizer, seq_len=201):
    if imp_array.ndim > 2:
        imp_array = np.mean(imp_array, axis=1)

    batch_size = imp_array.shape[0]
    bp_att = np.zeros((batch_size, seq_len))
    input_ids_list = input_ids_tensor.cpu().numpy()
    
    for i in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list[i])
        seq_idx = 0
        for t_idx, token in enumerate(tokens):
            if token in ['[CLS]', '<cls>', '<s>', '[PAD]', '<pad>', '[SEP]', '</s>']:
                continue
                
            clean_token = token.replace(' ', '').replace('Ġ', '')
            if clean_token in ['[UNK]', '<unk>']:
                clean_token = 'N' 
                
            t_len = len(clean_token)
            if t_len == 0:
                continue
                
            end_idx = min(seq_idx + t_len, seq_len)
            actual_len = end_idx - seq_idx
            
            if actual_len > 0:
                bp_att[i, seq_idx : end_idx] = imp_array[i, t_idx]
                seq_idx = end_idx
                
            if seq_idx >= seq_len:
                break
    return bp_att

def output_model_batch_inference(batch, model, tokenizer, device, rc=True):
    if isinstance(batch['sequence'], tuple):
        sequences_fw = list(batch['sequence'])
    else:
        sequences_fw = batch['sequence']
        
    tokens_fw = tokenizer(
        sequences_fw, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    input_ids_fw = tokens_fw['input_ids'].to(device)
    attention_mask_fw = tokens_fw['attention_mask'].to(device)
    output_fw, importance_fw = model(input_ids_fw, attention_mask_fw)

    if rc:
        if isinstance(batch['sequence_rev'], tuple):
            sequences_rc = list(batch['sequence_rev'])
        else:
            sequences_rc = batch['sequence_rev']
            
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
        output_rc = output_fw
        importance_rc = importance_fw
        input_ids_rc = input_ids_fw

    return output_fw, output_rc, importance_fw, importance_rc, input_ids_fw, input_ids_rc

def predict_only(model, dataloader, tokenizer, device, threshold=0.5, return_importance=True):
    model.eval()
    all_probs = []
    all_importance = []
    
    print(f"[*] Avvio inferenza su device: {device} (Tokenizzazione on-the-fly)")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            out_fw, out_rc, imp_fw, imp_rc, ids_fw, ids_rc = output_model_batch_inference(
                batch, model, tokenizer, device, rc=True
            )
            
            probs = torch.sigmoid((out_fw + out_rc) / 2)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            
            if return_importance and imp_fw is not None and imp_rc is not None:
                bp_att_fw = map_attention_to_bp(imp_fw.cpu().numpy(), ids_fw, tokenizer)
                bp_att_rc = map_attention_to_bp(imp_rc.cpu().numpy(), ids_rc, tokenizer)
                
                bp_att_rc_flipped = np.flip(bp_att_rc, axis=1)
                sym_bp_att = (bp_att_fw + bp_att_rc_flipped) / 2
                
                k = 6
                num_bins = int(np.ceil(201 / k))
                binned_att = np.zeros((imp_fw.shape[0], num_bins))
                
                for i in range(imp_fw.shape[0]):
                    for b in range(num_bins):
                        start = b * k
                        end = min((b + 1) * k, 201)
                        binned_att[i, b] = np.mean(sym_bp_att[i, start:end])
                        
                all_importance.extend([row for row in binned_att])

            del out_fw, out_rc, probs
            if imp_fw is not None: 
                del imp_fw, imp_rc

    probs_np = np.array(all_probs)
    preds_np = (probs_np >= threshold).astype(int)
    
    result = {'probabilities': probs_np, 'predictions': preds_np}
    if return_importance:
        result['importance'] = all_importance
        
    return result

def plot_attention_only(sample_idx, dataset, matrices, output_dir, label_type="MASK", k=6):
    """
    Aggiunto parametro label_type per differenziare i nomi dei file (es. TP_Positivo, TN_Negativo)
    """
    if isinstance(dataset, pd.DataFrame):
        seq = dataset.iloc[sample_idx]['sequence']
    else:
        seq = dataset[sample_idx]['sequence']
        
    attn_values = matrices[sample_idx] 
    num_bins = len(attn_values)
    
    tokens = []
    colors = []
    
    for b in range(num_bins):
        start = b * k
        end = min((b + 1) * k, len(seq))
        token_str = seq[start:end]
        tokens.append(token_str)
        
        if all(char == 'N' for char in token_str):
            colors.append('#CCCCCC') 
        else:
            colors.append('#006B7D') 
            
    df_plot = pd.DataFrame({
        'Index': range(len(tokens)),
        'Token': tokens,
        'Attention': attn_values,
        'Color': colors
    })

    plt.figure(figsize=(20, 8))
    
    ax = sns.barplot(
        data=df_plot, 
        x='Index', 
        y='Attention', 
        palette=colors, 
        legend=False
    )
    
    ax.set_xticklabels(df_plot['Token'])
    
    ax.set_ylabel('Attention Score', fontsize=18, fontweight='bold')
    ax.set_xlabel('Sequence Tokens (6-mers)', fontsize=18, fontweight='bold')
    if label_type == "TP_Positive":
        plt.title(f'Attention Map (True Positive)', fontsize=22, fontweight='bold', pad=20)
    else:
        plt.title(f'Attention Map (True Negative)', fontsize=22, fontweight='bold', pad=20)
    plt.xticks(rotation=90, ha='center', fontfamily='monospace', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"attention_sample_{sample_idx}_{label_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Plot attenzione salvato in: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for CadmusDNA")
    parser.add_argument("--dataset", type=str, required=True, help="Path al file dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path al file pesi")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output dir")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Caricamento dataset da: {args.dataset}")
    dataset = pd.read_pickle(args.dataset)
    
    if isinstance(dataset, pd.DataFrame):
        dataset_list = dataset.to_dict('records')
    else:
        dataset_list = dataset
    
    print("[*] Applicazione dello structural masking simmetrico (sostituzione bordi con 'N')...")
    for d_entry in dataset_list:
        seq_fw = d_entry['sequence']
        masked_fw = seq_fw[:18] + ('N' * 18) + seq_fw[36:162] + ('N' * 18) + seq_fw[180:]
        d_entry['sequence'] = masked_fw
        d_entry['sequence_rev'] = str(Seq(masked_fw).reverse_complement())

    from torch.utils.data import Dataset
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    test_loader = DataLoader(SimpleDataset(dataset_list), batch_size=args.batch_size, shuffle=False)

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

    model_name = "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = predict_only(model, test_loader, tokenizer, device)

    print("[*] Salvataggio dei risultati in corso...")
    df_results = pd.DataFrame({
        'probabilities': results['probabilities'],
        'predictions': results['predictions']
    })
    
    if 'label' in dataset_list[0]:
        df_results['true_label'] = [d['label'] for d in dataset_list]
        
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[*] Predizioni salvate in: {csv_path}")

    if 'importance' in results:
        flat_importance = results['importance'] 
                
        matrices_path = os.path.join(args.output_dir, "attention_matrices.pkl")
        with open(matrices_path, 'wb') as f:
            pickle.dump(flat_importance, f)
        print(f"[*] Matrici di attenzione salvate in: {matrices_path}")

        # --- RICERCA DEI VERI POSITIVI E VERI NEGATIVI ---
        if 'true_label' in df_results.columns:
            # Troviamo gli indici dove predizione e etichetta reale coincidono
            tp_indices = df_results[(df_results['predictions'] == 1) & (df_results['true_label'] == 1)].index.tolist()
            tn_indices = df_results[(df_results['predictions'] == 0) & (df_results['true_label'] == 0)].index.tolist()
            
            if tp_indices:
                print(f"[*] Plotto un esempio di Vero Positivo (Indice: {tp_indices[0]})")
                plot_attention_only(
                    sample_idx=tp_indices[0], 
                    dataset=dataset_list, 
                    matrices=flat_importance, 
                    output_dir=args.output_dir,
                    label_type="TP_Positive"
                )
            else:
                print("[!] Nessun Vero Positivo trovato.")

            if tn_indices:
                print(f"[*] Plotto un esempio di Vero Negativo (Indice: {tn_indices[0]})")
                plot_attention_only(
                    sample_idx=tn_indices[0], 
                    dataset=dataset_list, 
                    matrices=flat_importance, 
                    output_dir=args.output_dir,
                    label_type="TN_Negative"
                )
            else:
                print("[!] Nessun Vero Negativo trovato.")
        else:
            # Fallback nel caso in cui non ci siano labels nel dataset (es. puro test cieco)
            if len(dataset_list) > 0:
                plot_attention_only(0, dataset_list, flat_importance, args.output_dir)

    print("[*] Processo completato con successo!")

if __name__ == "__main__":
    main()