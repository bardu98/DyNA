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
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc



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
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0,:]
        out = self.final_linear(out)

        return torch.squeeze(out), attention_matrix


class CadmusDNA(nn.Module):
    
    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)
        
        

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




#DATASET

with open('../data/dataset_nup1_sapiens.pkl', 'rb') as f:  
    dataset_sapiens = pickle.load(f)
with open('../data/dataset_nup1_sapiens_RC.pkl', 'rb') as f:
    dataset_sapiens_rev = pickle.load(f)

RC_augmentation=True

if RC_augmentation:
    for d1, d2 in zip(dataset_sapiens, dataset_sapiens_rev):
        d1['embedding_rev'] = d2['embedding']  
        
data = dataset_sapiens

labels = [entry['label'] for entry in data]
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")



import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, recall_score, accuracy_score, 
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)




#FINE CARICAMENTO DATI


from sklearn.metrics import matthews_corrcoef


def set_seed(seed=42):
    random.seed(seed)                          # Seed Python (modulo random)
    np.random.seed(seed)                       # Seed NumPy
    torch.manual_seed(seed)                    # Seed PyTorch CPU
    torch.cuda.manual_seed(seed)               # Seed PyTorch CUDA (una GPU)
    torch.cuda.manual_seed_all(seed)           # Seed PyTorch CUDA (tutte le GPU)

    torch.backends.cudnn.deterministic = True  # Usa algoritmi deterministici
    torch.backends.cudnn.benchmark = False     # Disabilita ottimizzazioni non deterministiche

    print(f"Seeds fixed with seed = {seed}")
set_seed()

# -----------------------------------------------------------------
# PARTE 1: IMPORTAZIONI E DEFINIZIONE 'OBJECTIVE' PER OPTUNA
# -----------------------------------------------------------------
import optuna
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
import copy # Importato per 'local_data'
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

# --- OBJECTIVE PER OPTUNA ---


g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def objective(trial):
    """
    Funzione obiettivo che ottimizza sul VALIDATION SET INTERNO (5%).
    Salva anche le metriche del TEST SET esterno su Optuna per analisi post-hoc.
    """
    
    # --- Iperparametri ---
    hp_dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    hp_num_heads = trial.suggest_categorical('num_heads', [4, 5, 8]) 
    hp_batch_size = trial.suggest_categorical('batch_size', [32, 64])
    hp_lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    hp_weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5, log=True)
    hp_patience = trial.suggest_int('patience', 5, 15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_embedding = 2560
    
    # --- Preparazione Dati ---
    local_data = [x for x in data if x['embedding'].shape[0] == 28]

    if not local_data:
        return -1.0 

    dataset = Nuc_Dataset(local_data, dim_embedding)
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    if len(labels) == 0:
        return -1.0 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = {
        'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 
        'F1': [], 'AUROC': [], 'PR_AUC': []
    }
    test_fold_metrics = {
        'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 
        'F1': [], 'AUROC': [], 'PR_AUC': []
    }

    # --- K-Fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
        train_labels = labels[train_idx]
        
        # (95% Train / 5% Internal Val)
        internal_train_idx, internal_val_idx = train_test_split(
            train_idx,
            test_size=0.05,
            stratify=train_labels,
            random_state=fold
        )

        internal_train_subset = Subset(dataset, internal_train_idx)
        internal_val_subset = Subset(dataset, internal_val_idx)
        
        dataloader_internal_train = DataLoader(internal_train_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        dataloader_internal_val = DataLoader(internal_val_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        
        # Test loader del fold
        test_subset_cv = Subset(dataset, val_idx) 
        dataloader_test_cv = DataLoader(test_subset_cv, batch_size=hp_batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
        # Setup Modello
        transf_parameters_att = {'input_dim': dim_embedding, 
                                 'dropout_rate': hp_dropout_rate, 
                                 'num_heads': hp_num_heads}
        
        if 0 in label_counts and 1 in label_counts:
             pos_weight_val = label_counts[0] / label_counts[1]
        else:
             pos_weight_val = 1.0
             
        weights_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        
        model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
        
        # --- training ---
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
            lr=hp_lr,
            weight_decay=hp_weight_decay,
            patience=hp_patience
        )

        # --- CALCOLO METRICHE (Loop su Validation e Test) ---
        evaluations = [
            (val_labels, val_probs, fold_metrics),          # Set 1: Validation
            (test_labels, final_test_probs, test_fold_metrics) # Set 2: Test
        ]

        for y_true_raw, y_probs_raw, target_dict in evaluations:
            if len(y_true_raw) > 0:
                y_true = np.array(y_true_raw)
                y_probs = np.array(y_probs_raw)
                y_pred = (y_probs > 0.5).astype(int)

                # Confusion Matrix
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                except ValueError:
                    tn, fp, fn, tp = 0, 0, 0, 0 

                # Calcoli
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

                # Accumulo
                target_dict['Sn'].append(sn)
                target_dict['Sp'].append(sp)
                target_dict['ACC'].append(acc)
                target_dict['MCC'].append(mcc)
                target_dict['F1'].append(f1)
                target_dict['AUROC'].append(auroc)
                target_dict['PR_AUC'].append(pr_auc)
            else:
                for k in target_dict: target_dict[k].append(-1.0)

        # --- Pruning su MCC (Validation!) ---
        trial.report(np.mean(fold_metrics['MCC']), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- 4. Salvataggio User Attrs (Validation e Test) ---
    mean_val_mcc = np.mean(fold_metrics['MCC'])
    
    # Save Validation metrics
    for metric_name, values in fold_metrics.items():
        trial.set_user_attr(f"internal_mean_{metric_name}", np.mean(values))
        trial.set_user_attr(f"internal_std_{metric_name}", np.std(values))

    # Salve Test metrics 
    for metric_name, values in test_fold_metrics.items():
        trial.set_user_attr(f"test_mean_{metric_name}", np.mean(values))
        trial.set_user_attr(f"test_std_{metric_name}", np.std(values))

    # optimized optuna on vaidation (no test!)
    return mean_val_mcc

# -----------------------------------------------------------------
# PARTE 2: ESECUZIONE OPTUNA
# -----------------------------------------------------------------

print("--- AVVIO STUDIO OPTUNA ---")
seeded_sampler = TPESampler(seed=42) 


study = optuna.create_study(
    study_name='optuna_results',
    storage='sqlite:///optuna_results.db',
    load_if_exists=True,             
    direction='maximize',
    sampler=seeded_sampler,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)


study.optimize(objective, n_trials=100) 

print("\n--- STUDIO OPTUNA COMPLETATO ---")
print(f"Miglior trial (Mean Internal Val MCC): {study.best_value}")
print("Iperparametri migliori:")
print(study.best_params)

best_hyperparameters = study.best_params



