
import os
import sys
import copy
import math
import pickle
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  

sys.path.append(os.path.abspath('../src'))

from data_class import Nuc_Dataset
from utils import test_classification, training_validation_and_test_loop_classification
from model import CadmusDNA, TransformerNuc_Cadmus

# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seeds fixed with seed = {seed}")

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

def invert_attention(A):
    L = A.shape[0]
    assert A.shape[0] == A.shape[1], "La matrice deve essere quadrata"
    perm = [0] + list(range(L-1, 0, -1))
    A_corrected = A[np.ix_(perm, perm)]
    return A_corrected

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def applica_mascheramento(dataset):
    """Sostituisce con 'N' gli indici [18:36] e [162:180]"""
    for d in dataset:
        seq = d['sequence']
        masked_seq = seq[:18] + ('N' * 18) + seq[36:162] + ('N' * 18) + seq[180:]
        d['sequence'] = masked_seq
    return dataset
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Script di Inferenza e Estrazione Matrici")
    parser.add_argument("--dataset", type=str, choices=['lympho', 'act', 'rest'], required=True, 
                        help="Scegli il dataset su cui effettuare l'inferenza")
    parser.add_argument("--n_folds", type=int, required=True, help="Scegli il numero di fold")
    args = parser.parse_args()

    set_seed(42)

    # 1. Mappatura file e nomi di output
    file_map = {
        'lympho': '../data/data_pkl/Lymphoblastoid_99_8_percentile.pkl',
        'act': '../data/data_pkl/CD4T_h19_Act_tot_99_8_percentile.pkl',
        'rest': '../data/data_pkl/CD4T_h19_Rest_tot_99_8_percentile.pkl'
    }
    short_name_map = {'lympho': 'lymp', 'act': 'act', 'rest': 'rest'}
    full_name_map = {'lympho': 'lymphoblastoid', 'act': 'act', 'rest': 'rest'}

    data_path = file_map[args.dataset]
    short_name = short_name_map[args.dataset]
    full_name = full_name_map[args.dataset]

    # 2. Caricamento del SINGOLO dataset selezionato
    print(f"\n[*] Caricamento dataset: {full_name.upper()}")
    with open(data_path, 'rb') as f:   
        dataset_raw = pickle.load(f)

    dataset_raw = applica_mascheramento(dataset_raw)

    # RC Augmentation con Progress Bar
    print("[*] Applicazione Reverse Complement...")
    for d in tqdm(dataset_raw, desc="RC Augmentation", unit="seq"):
        d['sequence_rev'] = str(Seq(d['sequence']).reverse_complement())

    print(f"\nLunghezza dataset {full_name}: {len(dataset_raw)}")

    print(f"--- {full_name.upper()} Label Counts ---")
    labels_list = [entry['label'] for entry in dataset_raw]
    for label, count in Counter(labels_list).items():
        print(f"Label {label}: {count} samples")

    # 3. Setup Modello e Dataloader
    best_hyperparameters = {
        'dropout_rate': 0.3832290311184182, 
        'num_heads': 5,
        'batch_size': 32, 
        'lr': 2.3270677083837795e-06,
        'weight_decay': 8.179499475211672e-08, 
        'patience': 10
    }

    dim_embedding = 2560
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = torch.Generator()
    g.manual_seed(0)

    transf_parameters_att = {
        'input_dim': dim_embedding, 
        'dropout_rate': best_hyperparameters['dropout_rate'], 
        'num_heads': best_hyperparameters['num_heads']
    }   

    dataset_obj = Nuc_Dataset(dataset_raw, max_length=67, rc_augmentation=True)
    
    dataloader_infer = DataLoader(dataset_obj, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g, drop_last=False)

    print(f"\n[*] inference on {args.n_folds} Folds...")
    
    for fold in tqdm(range(args.n_folds), desc="Processing Folds", unit="fold"):
        
        best_model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
        state_dict = torch.load(f'best_model_weights_99_8_percentile_fold{fold}_MASKED_07_03_26.pt', map_location=device)#(f"best_model_weights_99_8_percentile_fold{fold}_03_03_26.pt", map_location=device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)
        best_model.eval()
        
        metrics, val_labels, val_preds, importance, importance_rc, preds = test_classification(best_model, dataloader_infer, threshold=0.5)
        
        tqdm.write(f"   -> Metriche Fold {fold}: MCC={metrics['MCC']:.3f}, AUC={metrics['AUC']:.3f}")
        
        nome_file_preds = f"../results/preds_{short_name}_model_fold{fold}_08_03_26_MASK.pkl"
        with open(nome_file_preds, 'wb') as file:
            pickle.dump(preds, file)
        
        all_matrices_dir = [m.cpu() if isinstance(m, torch.Tensor) else m for batch in importance for m in batch]
        all_matrices_rc = [m.cpu() if isinstance(m, torch.Tensor) else m for batch in importance_rc for m in batch]
        
        checkpoint_data = {
            'matrices_dir': all_matrices_dir,
            'matrices_rc': all_matrices_rc,
            'fold_index': fold
        }
        nome_file_matrici = f"../results/matrices_results_fold{fold}_{full_name}_MASK_08_02_2026.pt"  #save now in results directory
        torch.save(checkpoint_data, nome_file_matrici)

    # ==============================================================================
    #  (ENSEMBLE)
    # ==============================================================================
    print("\n[*] Inizio aggregazione predizioni (Ensemble)...")

    total_results = np.array(pd.read_pickle(f"../results/preds_{short_name}_model_fold0_08_03_26_MASK.pkl"))

    for i in tqdm(range(1, args.n_folds), desc="Aggregating Folds", unit="fold"):
        preds_list = pd.read_pickle(f"../results/preds_{short_name}_model_fold{i}_08_03_26_MASK.pkl")
        total_results += np.array(preds_list)

    final_predictions = total_results / args.n_folds

    print(f"\nLunghezza totale predizioni: {len(final_predictions)}")
    print("Predizioni finali (primi 5 elementi):", final_predictions[:5])

    nome_file_finale = f"../results/preds_{full_name}_model_sum_folds_08_03_26_MASK.pkl"
    with open(nome_file_finale, 'wb') as file:
        pickle.dump(final_predictions, file)

    print(f"\n✅ Operazione completata! Lista aggregata salvata in: {nome_file_finale}")

if __name__ == "__main__":
    main()