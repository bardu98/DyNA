import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import sys
sys.path.append(os.path.abspath('../src'))
from data_class import Nuc_Dataset
from utils import training_validation_and_test_loop_classification
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio.Seq import Seq
import random
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc

from sklearn.metrics import (confusion_matrix, recall_score, accuracy_score, 
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, f1_score)


# -----------------------------------------------------------------
# MODEL CLASSES
# -----------------------------------------------------------------
class TransformerNuc_DyNA(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout_rate=0.0, 
                 f_activation=nn.ReLU()):
        super(TransformerNuc_DyNA, self).__init__()

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
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0,:]
        out = self.final_linear(out)

        return torch.squeeze(out), attention_matrix


class DyNA(nn.Module):
    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.attention_model = att_module(**att_parameters).to(self.device)

    def forward(self, seqs):
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


# -----------------------------------------------------------------
# SEEDS
# -----------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    print(f"Seeds fixed with seed = {seed}")

set_seed()
g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------
# DATA
# -----------------------------------------------------------------
with open('../data/dataset_nup1_sapiens.pkl', 'rb') as f:  
    dataset_sapiens = pickle.load(f)

# RC_augmentation = False, WITHOUT SIAMESE
data = dataset_sapiens

labels = [entry['label'] for entry in data]
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")


# -----------------------------------------------------------------
# CROSS-VALIDATION 
# -----------------------------------------------------------------
def run_evaluation():
    best_hyperparameters = {
        'dropout_rate': 0.3832290311184182, 
        'num_heads': 5,
        'batch_size': 32, 
        'lr': 2.3270677083837795e-06,
        'weight_decay': 8.179499475211672e-08, 
        'patience': 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_embedding = 2560
    
    local_data = [x for x in data if x['embedding'].shape[0] == 28]
    if not local_data:
        print("Nessun dato corrispondente trovato.")
        return

    dataset = Nuc_Dataset(local_data, dim_embedding)
    labels_array = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = {'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 'F1': [], 'AUROC': [], 'PR_AUC': []}
    test_fold_metrics = {'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 'F1': [], 'AUROC': [], 'PR_AUC': []}

    print("\n--- INIZIO 5-FOLD CROSS VALIDATION ---")
    print(f"Hyperparameters: {best_hyperparameters}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_array)), labels_array)):
        print(f"=== FOLD {fold + 1}/5 ===")
        
        train_labels_fold = labels_array[train_idx]
        
        # Split 95% Train / 5% Internal Validation
        internal_train_idx, internal_val_idx = train_test_split(
            train_idx,
            test_size=0.05,
            stratify=train_labels_fold,
            random_state=fold
        )

        internal_train_subset = Subset(dataset, internal_train_idx)
        internal_val_subset = Subset(dataset, internal_val_idx)
        test_subset_cv = Subset(dataset, val_idx) 
        
        # Dataloaders
        bs = best_hyperparameters['batch_size']
        dataloader_internal_train = DataLoader(internal_train_subset, batch_size=bs, shuffle=True, worker_init_fn=seed_worker, generator=g)
        dataloader_internal_val = DataLoader(internal_val_subset, batch_size=bs, shuffle=True, worker_init_fn=seed_worker, generator=g)
        dataloader_test_cv = DataLoader(test_subset_cv, batch_size=bs, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
        # Model
        transf_parameters_att = {
            'input_dim': dim_embedding, 
            'dropout_rate': best_hyperparameters['dropout_rate'], 
            'num_heads': best_hyperparameters['num_heads']
        }
        
        model_internal = DyNA(TransformerNuc_DyNA, transf_parameters_att, device)
        
        # Training
        (train_mcc_list, val_mcc_list, loss_train, loss_val,
         best_val_loss, best_state_cpu, epoch_best,
         _dict_probs, _dict_labels, 
         val_labels, val_probs, test_labels, final_test_probs, 
         best_val_probs, best_true_val) = training_validation_and_test_loop_classification(
            model_internal,
            dataloader_internal_train,
            dataloader_internal_val, 
            dataloader_test_cv,     
            epochs=200,             
            lr=best_hyperparameters['lr'],
            weight_decay=best_hyperparameters['weight_decay'],
            patience=best_hyperparameters['patience']
        )

        # metrics
        evaluations = [
            (val_labels, val_probs, fold_metrics, "VALIDATION"),
            (test_labels, final_test_probs, test_fold_metrics, "TEST SET")
        ]

        for y_true_raw, y_probs_raw, target_dict, phase_name in evaluations:
            if len(y_true_raw) > 0:
                y_true = np.array(y_true_raw)
                y_probs = np.array(y_probs_raw)
                y_pred = (y_probs > 0.5).astype(int)

                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                except ValueError:
                    tn, fp, fn, tp = 0, 0, 0, 0 

                sn = recall_score(y_true, y_pred, zero_division=0) 
                sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0      
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                try:
                    auroc = roc_auc_score(y_true, y_probs)
                    pr_auc = average_precision_score(y_true, y_probs)
                except ValueError:
                    auroc, pr_auc = 0.5, 0.0 

                target_dict['Sn'].append(sn)
                target_dict['Sp'].append(sp)
                target_dict['ACC'].append(acc)
                target_dict['MCC'].append(mcc)
                target_dict['F1'].append(f1)
                target_dict['AUROC'].append(auroc)
                target_dict['PR_AUC'].append(pr_auc)
                
                if phase_name == "TEST SET":
                    print(f"   Test Set -> MCC: {mcc:.4f} | AUROC: {auroc:.4f} | ACC: {acc:.4f}")
            else:
                for k in target_dict: target_dict[k].append(-1.0)
                
    # Final results
    print("\n" + "="*50)
    print("FINAL RESULTS(Mean ± std on 5 Folds)")
    print("="*50)
    
    print("\n[ VALIDATION]")
    for metric_name in fold_metrics:
        mean_val = np.mean(fold_metrics[metric_name])
        std_val = np.std(fold_metrics[metric_name])
        print(f"  {metric_name}:\t{mean_val:.4f} ± {std_val:.4f}")

    print("\n[TEST]")
    for metric_name in test_fold_metrics:
        mean_val = np.mean(test_fold_metrics[metric_name])
        std_val = np.std(test_fold_metrics[metric_name])
        print(f"  {metric_name}:\t{mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    run_evaluation()