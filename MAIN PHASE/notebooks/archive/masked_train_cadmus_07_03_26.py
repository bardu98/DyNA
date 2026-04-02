import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import sys
import gc

# Aggiunge la cartella src al path
sys.path.append(os.path.abspath('../src'))

# Import personalizzati
from data_class import Nuc_Dataset
from utils import test_classification, training_validation_and_test_loop_classification
from model import CadmusDNA, TransformerNuc_Cadmus

from collections import Counter
from sklearn.model_selection import StratifiedKFold
import math
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio.Seq import Seq
import random
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score

RC_augmentation = True

# =====================================================================
# 1. CARICAMENTO DATI (Solo Train)
# =====================================================================
with open('../data/data_pkl/Lymphoblastoid_99_8_percentile.pkl', 'rb') as f: 
    dataset_train_raw = pickle.load(f)

# =====================================================================
# 2. MASCHERAMENTO E REVERSE COMPLEMENT
# =====================================================================
def applica_mascheramento(dataset):
    """Sostituisce con 'N' gli indici [18:36] e [162:180]"""
    for d in dataset:
        seq = d['sequence']
        masked_seq = seq[:18] + ('N' * 18) + seq[36:162] + ('N' * 18) + seq[180:]
        d['sequence'] = masked_seq
    return dataset

dataset_train_raw = applica_mascheramento(dataset_train_raw)

if RC_augmentation:
    for d in dataset_train_raw: 
        d['sequence_rev'] = str(Seq(d['sequence']).reverse_complement())

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seeds fixed with seed = {seed}")
set_seed()

labels = [entry['label'] for entry in dataset_train_raw]

best_hyperparameters ={'dropout_rate': 0.3832290311184182, 'num_heads': 5,
                       'batch_size': 32, 'lr': 2.3270677083837795e-06,
                       'weight_decay': 8.179499475211672e-08, 'patience': 10}

def classification_metrics(y_true, probs_pred, threshold=0.5):
    y_pred = (np.array(probs_pred) >= threshold).astype(int)
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, probs_pred)
    return {
        "Sensitivity_val": sensitivity,
        "Specificity_val": specificity,
        "Accuracy_val": accuracy,
        "MCC_val": mcc,
        "AUC_val": auc
    }

g = torch.Generator()
g.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_embedding = 2560
labels = np.array(labels)

num_workers = 8
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# =====================================================================
# 3. CREAZIONE DATALOADER
# =====================================================================
# max_length=80 fondamentale per non tagliare la sequenza frammentata dalle N
dataset_train_obj = Nuc_Dataset(dataset_train_raw, max_length=67, rc_augmentation=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
    print(f"\n--- Fold {fold+1} ---")

    train_subset_cv = Subset(dataset_train_obj, train_idx) 
    val_subset_cv = Subset(dataset_train_obj, val_idx)
    
    # --- IL TRUCCO DEL DUMMY TEST SET ---
    # Prendiamo i primi DUE indici per evitare che lo .squeeze() di PyTorch faccia collassare la dimensione della batch
    dummy_test_subset = Subset(dataset_train_obj, val_idx[:2])

    dataloader_train_cv = DataLoader(
        train_subset_cv, batch_size=32, shuffle=True,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=num_workers, 
        pin_memory=True, persistent_workers=True
    )
    
    dataloader_val_cv = DataLoader(
        val_subset_cv, batch_size=32, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=num_workers, 
        pin_memory=True, persistent_workers=True
    )

    # Dataloader per il dummy test: batch_size=2, num_workers=0 per zero overhead
    dataloader_dummy_test = DataLoader(
        dummy_test_subset, batch_size=2, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
        drop_last=False, num_workers=0, 
        pin_memory=True
    )

    transf_parameters_att = {'input_dim': dim_embedding, 
                             'dropout_rate': best_hyperparameters['dropout_rate'], 
                             'num_heads': best_hyperparameters['num_heads']}   

    model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)

    # Passiamo il dataloader_dummy_test come blind test set
    _, _, _, _, best_val_acc, best_state_cpu, best_epoch, _, _, val_labels, val_probs, test_labels, test_probs, best_val_probs, best_true_val = training_validation_and_test_loop_classification(
        model_internal,
        dataloader_train_cv,
        dataloader_val_cv,
        dataloader_dummy_test,
        epochs= 200,
        lr=best_hyperparameters['lr'],
        weight_decay=best_hyperparameters['weight_decay'],
        patience=best_hyperparameters['patience']
    )

    save_path = f"best_model_weights_99_8_percentile_fold{fold}_MASKED_07_03_26.pt"
    torch.save(best_state_cpu, save_path)

    # Calcoliamo le metriche usando le probabilità salvate nell'epoca migliore
    metrics_val = classification_metrics(best_true_val, best_val_probs)
    fold_results.append({
        'fold': fold + 1,
        'best_epoch': best_epoch
    })
    fold_results[-1].update(metrics_val)
    
    print(f"[*] Fold {fold+1} completato (Best Epoch: {best_epoch}) - Validation AUC: {metrics_val['AUC_val']:.4f}")

    # --- LIBERAZIONE MEMORIA ---
    del model_internal, dataloader_train_cv, dataloader_val_cv, dataloader_dummy_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- STAMPA FINALE ---
print("\n========== RISULTATI FINALI IN CROSS-VALIDATION (INTERNAL VALIDATION) ==========")
if fold_results:
    for metrics in ["Sensitivity_val", "Specificity_val", "Accuracy_val", "MCC_val", "AUC_val"]:
        print(f"{metrics}: {np.mean([i[metrics] for i in fold_results]):.4f}")
else:
    print("Nessun fold eseguito.")