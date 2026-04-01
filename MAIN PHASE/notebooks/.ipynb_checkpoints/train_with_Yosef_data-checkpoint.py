import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import sys
import gc # <--- AGGIUNTO per liberare memoria
# Aggiunge la cartella src al path
sys.path.append(os.path.abspath('../src'))
# Ora puoi importare normalmente
from data_class import Nuc_Dataset
from utils import test_classification,training_validation_and_test_loop_classification
from model import CadmusDNA,TransformerNuc_Cadmus
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio.Seq import Seq
import random
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score


RC_augmentation = True

with open('../../INFERENCE/NuPoSe_Yosef_Data/Lympho_Nucleosomal_TOT.pkl', 'rb') as f: 
    dataset_sapiens = pickle.load(f)

if RC_augmentation:
    for d1 in dataset_sapiens: 
        d1['sequence_rev'] = str(Seq(d1['sequence']).reverse_complement())

data_train = dataset_sapiens

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

labels = [entry['label'] for entry in data_train]
label_counts = Counter(labels)

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
labels = np.array([entry['label'] for entry in data_train])

num_workers=8
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

dataset_train_obj = Nuc_Dataset(data_train, max_length=37, rc_augmentation=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
    print(f"\n--- Fold {fold+1} ---")

    train_subset_cv = Subset(dataset_train_obj, train_idx) 
    test_subset_cv = Subset(dataset_train_obj, val_idx)
    
    # Sample molto piccolo della validation per velocizzare
    sample_size = min(len(val_idx), 200)
    small_indices = np.random.choice(val_idx, sample_size, replace=False)
    small_test_subset = Subset(dataset_train_obj, small_indices)

    dataloader_train_cv = DataLoader(
        train_subset_cv, batch_size=32, shuffle=True,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=num_workers, 
        pin_memory=True, persistent_workers=True
    )
    
    dataloader_test_cv = DataLoader(
        test_subset_cv, batch_size=32, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=num_workers, 
        pin_memory=True, persistent_workers=True
    )

    dataloader_test_cv_resting = DataLoader(
        small_test_subset, batch_size=32, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=num_workers, 
        pin_memory=True, persistent_workers=True
    )

    transf_parameters_att = {'input_dim': dim_embedding, 
                             'dropout_rate': best_hyperparameters['dropout_rate'], 
                             'num_heads': best_hyperparameters['num_heads']}   

    model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)

    _, _, _, _, best_val_acc, best_state_cpu, best_epoch, _, _, val_labels, val_probs, test_labels, test_probs, best_val_probs, best_true_val = training_validation_and_test_loop_classification(
        model_internal,
        dataloader_train_cv,
        dataloader_test_cv,
        dataloader_test_cv_resting, 
        epochs= 200,
        lr=best_hyperparameters['lr'],
        weight_decay=best_hyperparameters['weight_decay'],
        patience=best_hyperparameters['patience']
    )

    save_path = f"YOSEF_best_model_weights_99_8_percentile_fold{fold}_03_03_26.pt"
    torch.save(best_state_cpu, save_path)

    best_model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    state_dict = torch.load(save_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)
    best_model.eval()
    
    fold_results.append({
        'fold': fold + 1,
        'test_metrics': test_classification(best_model, dataloader_test_cv_resting, threshold=0.5)[0],
        'best_epoch': best_epoch
    })
    fold_results[-1].update(classification_metrics(best_true_val, best_val_probs))

    # --- LIBERAZIONE MEMORIA ---
    del model_internal, best_model, dataloader_train_cv, dataloader_test_cv, dataloader_test_cv_resting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- STAMPA FINALE ---
if fold_results:
    for metrics in ["Sensitivity_val", "Specificity_val", "Accuracy_val", "MCC_val", "AUC_val"]:
        print(metrics, np.mean([i[metrics] for i in fold_results]))
else:
    print("Nessun fold eseguito.")