import pandas as pd
import pickle
import torch
import os
import sys
import numpy as np
import random
import argparse 
import math
import copy
from collections import Counter, defaultdict

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.samplers import TPESampler

sys.path.append(os.path.abspath('../src'))
from data_class import Nuc_Dataset
from utils import training_validation_and_test_loop_classification 
from model import CadmusDNA, TransformerNuc_Cadmus
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, 
                             confusion_matrix, matthews_corrcoef, f1_score, 
                             average_precision_score)


# --- 1. SETUP ARGS and SEED ---
parser = argparse.ArgumentParser(description="Cadmus Hyperparameters")
parser.add_argument("data_df_path", type=str, help="Path dataset normale")
parser.add_argument("data_df_path_RC", type=str, help="Path dataset RC")
args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds fixed with seed = {seed}")

set_seed(42)

# --- 2. CARICAMENTO DATI ---
print(f"Load data from: {args.data_df_path}")
with open(args.data_df_path, 'rb') as f:   
    dataset_sapiens = pickle.load(f)

print(f"Load RC data from: {args.data_df_path_RC}")
with open(args.data_df_path_RC, 'rb') as f: 
    dataset_sapiens_rev = pickle.load(f)

for d1, d2 in zip(dataset_sapiens, dataset_sapiens_rev):
    d1['embedding_rev'] = d2['embedding']  

data = dataset_sapiens

labels = [entry['label'] for entry in data]
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")




#########################################
# FIND BEST HYPERPARAMETERS WITH OPTUNA #
#########################################


g = torch.Generator()
g.manual_seed(42) # Usa lo stesso seed

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def objective(trial):
    """
    Funzione obiettivo di Optuna con salvataggio metriche estese.
    """
    
    # --- 1. Definizione Iperparametri ---
    hp_dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    hp_num_heads = trial.suggest_categorical('num_heads', [4, 5, 8]) 
    hp_batch_size = trial.suggest_categorical('batch_size', [32, 64])
    hp_lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    hp_weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5, log=True)
    hp_patience = trial.suggest_int('patience', 5, 15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_embedding = 2560
    
    # --- 2. Filtro Dati ---
    local_data = [x for x in data if x['embedding'].shape[0] == 28]

    if not local_data:
        print("Attenzione: nessun dato rimasto dopo il filtraggio.")
        return -1.0 

    dataset = Nuc_Dataset(local_data, dim_embedding, truncate_last=False)
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    if len(labels) == 0:
        print("Attenzione: nessun label nel dataset.")
        return -1.0 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- [FIX 1] Inizializzazione Liste Metriche (PRIMA del loop) ---
    fold_metrics = {
        'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 
        'F1': [], 'AUROC': [], 'PR_AUC': []
    }

    # --- 3. Ciclo K-Fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
        train_labels = labels[train_idx]
        
        internal_train_idx, internal_val_idx = train_test_split(
            train_idx,
            test_size=0.05,
            stratify=train_labels,
            random_state=fold
        )

        internal_train_subset = Subset(dataset, internal_train_idx)
        internal_val_subset = Subset(dataset, internal_val_idx)
        
        # Dataloaders
        dataloader_internal_train = DataLoader(internal_train_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        dataloader_internal_val = DataLoader(internal_val_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        
        # Test loader del fold
        test_subset_cv = Subset(dataset, val_idx) 
        dataloader_test_cv = DataLoader(test_subset_cv, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        
        # Setup Modello
        transf_parameters_att = {'input_dim': dim_embedding, 
                                 'dropout_rate': hp_dropout_rate, 
                                 'num_heads': hp_num_heads}
        
        # Calcolo Pesi per la Loss (Binaria)
        # Assumiamo label_counts sia globale o calcolato sopra
        if 0 in label_counts and 1 in label_counts:
             pos_weight_val = label_counts[0] / label_counts[1]
        else:
             pos_weight_val = 1.0
             
        weights_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        
        model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
        
        # --- Allenamento ---
        # [FIX 2] Passiamo weight_tensor correttamente e recuperiamo i risultati giusti
        # Return signature attesa: 
        # ..., test_labels, test_probs, best_val_probs, best_true_val
        # Gli indici -4 e -3 corrispondono a test_labels e test_probs finali
        results = training_validation_and_test_loop_classification(
            model_internal,
            dataloader_internal_train,
            dataloader_internal_val, 
            dataloader_test_cv,      
            epochs=200,              
            lr=hp_lr,
            weight_decay=hp_weight_decay,
            weight_tensor=weights_tensor,  # Importante: passiamo il peso
            patience=hp_patience
        )

        test_labels = results[-4] 
        test_probs = results[-3]

        # --- [FIX 3] Calcolo Metriche (DENTRO il loop fold) ---
        if len(test_labels) > 0:
            y_true = np.array(test_labels)
            y_probs = np.array(test_probs)
            y_pred = (y_probs > 0.5).astype(int)

            # Confusion Matrix Elements
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
            fold_metrics['Sn'].append(sn)
            fold_metrics['Sp'].append(sp)
            fold_metrics['ACC'].append(acc)
            fold_metrics['MCC'].append(mcc)
            fold_metrics['F1'].append(f1)
            fold_metrics['AUROC'].append(auroc)
            fold_metrics['PR_AUC'].append(pr_auc)

        else:
            # Fallback in caso di fold vuoto o errore
            for k in fold_metrics: fold_metrics[k].append(-1.0)

        # --- Pruning su MCC ---
        trial.report(np.mean(fold_metrics['MCC']), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- 4. Calcolo Medie e Salvataggio (FUORI dal loop fold) ---
    mean_mcc = np.mean(fold_metrics['MCC'])
    
    for metric_name, values in fold_metrics.items():
        trial.set_user_attr(f"mean_{metric_name}", np.mean(values))
        trial.set_user_attr(f"std_{metric_name}", np.std(values))

    return mean_mcc
# --- FINE DEFINIZIONE OBJECTIVE ---


# -----------------------------------------------------------------
# PARTE 2: ESECUZIONE DELLO STUDIO OPTUNA
# -----------------------------------------------------------------

print("--- AVVIO STUDIO OPTUNA ---")
# Devi creare un sampler "seminato"
seeded_sampler = TPESampler(seed=42) # Usa lo stesso seed

study = optuna.create_study(
    # study_name='my_cv_optimization_cadmus_15_12_25_HS', 
    # storage='sqlite:///my_study_cadmus_15_12_25_H.db',  
    study_name='my_cv_optimization_cadmus_02_02_26_MEL', 
    storage='sqlite:///my_study_cadmus_02_02_26_MEL.db', 
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

# Salva i migliori parametri per la Parte 3
best_hyperparameters = study.best_params

print("\n" + "="*60)
print(f" RISULTATI MIGLIOR TRIAL (Trial #{study.best_trial.number})")
print("="*60)

best_trial = study.best_trial

# 1. Obiettivo Principale
print(f"\n Miglior MCC (Obiettivo): {best_trial.value:.4f}")

# 2. Iperparametri Ottimali
print("\n  Migliori Iperparametri:")
for key, value in best_trial.params.items():
    print(f"   - {key:<15}: {value}")

# 3. Tutte le Metriche Salvate (User Attributes)
print("\n Metriche Dettagliate (Media su 5-Fold):")

# Recuperiamo il dizionario degli attributi salvati
metrics = best_trial.user_attrs

# Le stampiamo ordinate alfabeticamente per chiarezza
keys = sorted(metrics.keys())

# Separiamo visivamente le medie dalle deviazioni standard (se presenti)
print("   --- Valori Medi ---")
for key in keys:
    if key.startswith("mean_"):
        print(f"   - {key[5:].upper():<10}: {metrics[key]:.4f}")

print("\n   --- Deviazioni Standard (Stabilità) ---")
for key in keys:
    if key.startswith("std_"):
        print(f"   - {key[4:].upper():<10}: {metrics[key]:.4f}")

print("="*60)

# Opzionale: Se vuoi salvare questi risultati in un dizionario per usarli dopo
best_results_dict = {
    "trial_number": best_trial.number,
    "best_mcc": best_trial.value,
    **best_trial.params,
    **best_trial.user_attrs
}


import json

# Creiamo un nome file basato sullo studio
results_filename = "results_optuna_MEL_02_02_26.json"

# Prepariamo il dizionario finale
final_save_data = {
    "study_name": study.study_name,
    "best_trial_number": study.best_trial.number,
    "best_params": study.best_params,
    "best_metrics_mean": {k: v for k, v in study.best_trial.user_attrs.items() if "mean" in k},
    "best_metrics_std": {k: v for k, v in study.best_trial.user_attrs.items() if "std" in k},
    "total_trials": len(study.trials)
}

# Salvataggio su file JSON (leggibile da un essere umano)
with open(results_filename, "w") as f:
    json.dump(final_save_data, f, indent=4)

print(f"\n[INFO] Risultati principali salvati in: {results_filename}")





# Salvataggio dell'intero dizionario dei risultati (incluso l'oggetto best_trial)
with open("best_model_details_MEL_02_02_26.pkl", "wb") as f:
    pickle.dump(best_results_dict, f)

# Se vuoi salvare lo studio Optuna intero per ricaricarlo in futuro
with open("optuna_study_full_MEL_02_02_26.pkl", "wb") as f:
    pickle.dump(study, f)

print("[INFO] Dettagli binari salvati in .pkl")







# # -----------------------------------------------------------------
# # PARTE 3: ESECUZIONE FINALE CON I MIGLIORI IPERPARAMETRI
# # (Questo è il tuo codice originale, modificato per usare i best_params)
# # -----------------------------------------------------------------

# print("\n--- AVVIO CROSS-VALIDATION FINALE CON HP OTTIMIZZATI ---")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dim_embedding = 2560
# # Rieseguiamo il filtraggio sui dati originali
# data_filtered = [x for x in data if x['embedding'].shape[0] == 28] 

# dataset = Nuc_Dataset(data_filtered, dim_embedding, drop_last=False)

# labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# fold_results = []
# for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#     print(f"\n--- Fold {fold+1} ---")

#     train_labels = labels[train_idx]
    
#     internal_train_idx, internal_val_idx = train_test_split(
#         train_idx,
#         test_size=0.05,
#         stratify=train_labels,
#         random_state=fold
#     )

#     internal_train_subset = Subset(dataset, internal_train_idx)
#     internal_val_subset = Subset(dataset, internal_val_idx)
    
#     # --- USA HP OTTIMIZZATI ---
#     dataloader_internal_train = DataLoader(internal_train_subset, batch_size=best_hyperparameters['batch_size'], shuffle=True)
#     dataloader_internal_val = DataLoader(internal_val_subset, batch_size=best_hyperparameters['batch_size'], shuffle=True)

#     test_subset_cv = Subset(dataset, val_idx) 
#     dataloader_test_cv = DataLoader(test_subset_cv, batch_size=best_hyperparameters['batch_size'], shuffle=True)
    
#     # --- USA HP OTTIMIZZATI ---
#     transf_parameters_att = {'input_dim': dim_embedding, 
#                              'dropout_rate': best_hyperparameters['dropout_rate'], 
#                              'num_heads': best_hyperparameters['num_heads']}
    
#     # Calcolo pesi (codice originale)
#     class_weights = {label: len(data_filtered) / count for label, count in label_counts.items()}
#     max_weight = max(class_weights.values())
#     class_weights = {label: weight / max_weight for label, weight in class_weights.items()}
#     weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
#     labels_list = ['label']
#     weight_dict = {label: weights_tensor for label in labels_list}
#     print(weight_dict)
    
#     model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    
#     # --- USA HP OTTIMIZZATI ---
#     # Allenamento finale: ora ci interessano le performance su dataloader_test_cv
#     _, _, _, _, best_val_acc, best_model, best_epoch, _, _, val_labels, val_probs, test_labels, test_probs = training_validation_and_test_loop_classification(
#         model_internal,
#         dataloader_internal_train,
#         dataloader_internal_val,
#         dataloader_test_cv, # True Test set
#         epochs= 200,
#         lr=best_hyperparameters['lr'],
#         weight_decay=best_hyperparameters['weight_decay'],
#         patience=best_hyperparameters['patience']
#     )

#     # Salva i risultati del fold (valutati sul test set del fold CV)
#     fold_results.append({
#         'fold': fold + 1,
#         'test_metrics': test_classification(best_model, dataloader_test_cv, threshold=0.5)[0],
#         'best_epoch': best_epoch
#     })

# # Stampa i risultati finali medi (sul test set dei 5 fold)
# print("\n--- RISULTATI FINALI CV (su test folds) CON HP OTTIMIZZATI ---")
# for metrics in ['Sensitivity (Recall)',
#   'Specificity',
#   'Accuracy',
#   'MCC',
#   'AUC',
#   'PR AUC',]:
#     # Gestisce il caso in cui la metrica non sia presente
#     metric_values = [i['test_metrics'].get(metrics) for i in fold_results if i['test_metrics'].get(metrics) is not None]
#     if metric_values:
#         print(f"{metrics}: {np.mean(metric_values)}")
#     else:
#         print(f"{metrics}: N/A (Metrica non trovata)")









    