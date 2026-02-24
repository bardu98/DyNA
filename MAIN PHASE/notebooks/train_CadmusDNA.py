import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import sys
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

with open('../data/data_pkl/Lymphoblastoid_99_8_percentile.pkl', 'rb') as f: 
    dataset_sapiens = pickle.load(f)

if RC_augmentation:
    for d1, d2 in zip(dataset_sapiens, dataset_sapiens):
        d1['sequence_rev'] = str(Seq(d1['sequence']).reverse_complement())

dataset_sapiens_resting = []
with open('../data/data_pkl/CD4T_h19_Rest_tot_99_8_percentile.pkl', 'rb') as f:
    dataset_sapiens_resting += pickle.load(f)

if RC_augmentation:
    for d1, d2 in zip(dataset_sapiens_resting, dataset_sapiens_resting):
        d1['sequence_rev'] = str(Seq(d1['sequence']).reverse_complement())

        
data_train = dataset_sapiens
data_test = dataset_sapiens_resting

len(data_test)


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

labels = [entry['label'] for entry in data_train]
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

# best_hyperparameters = {'dropout_rate': 0.3143462158665756,
#  'num_heads': 8,
#  'batch_size': 32,
#  'lr': 1.4271180164633878e-06,
#  'weight_decay': 1.2306663346888495e-06,
#  'patience': 3}#12}


best_hyperparameters ={'dropout_rate': 0.3832290311184182, 'num_heads': 5,
                       'batch_size': 32, 'lr': 2.3270677083837795e-06,
                       'weight_decay': 8.179499475211672e-08, 'patience': 10}


# best_hyperparameters = {'dropout_rate': 0.47805461769215474,
#                          'num_heads': 5,
#                          'batch_size': 64,
#                          'lr': 1.2764937047792502e-05,
#                          'weight_decay': 8.322724281704737e-07,
#                          'patience': 5}



# #CV WITH EARLY STOPPING

def classification_metrics(y_true, probs_pred, threshold=0.5):
    """
    Calcola Sensitivity, Specificity, Accuracy, MCC e AUC
    a partire da y_true (etichette vere) e probs_pred (probabilità predette).
    
    Parametri:
        y_true: array-like di etichette vere (0 o 1)
        probs_pred: array-like di probabilità predette (float tra 0 e 1)
        threshold: soglia di classificazione (default 0.5)
    
    Ritorna:
        dizionario con le metriche richieste
    """
    # Conversione in predizioni binarie
    y_pred = (np.array(probs_pred) >= threshold).astype(int)
    y_true = np.array(y_true)
    
    # Matrice di confusione
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metriche
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
g.manual_seed(0) # Usa lo stesso seed

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim_embedding = 2560


######################
#dataset_train = Nuc_Dataset(data_train, dim_embedding, drop_last=False)
dataset_train = Nuc_Dataset(data_train, max_length=201, rc_augmentation=True)
# dataset_test = Nuc_Dataset(data_test, dim_embedding, drop_last=False)
#labels = np.array([dataset_train[i]['label'].item() for i in range(len(dataset_train))])   #al posto di data_train c'era dataset_train

labels = np.array([entry['label'] for entry in data_train])
####################


##################
#labels = np.array([data_train[i]['label'].item() for i in range(len(data_train))])   #al posto di data_train c'era dataset_train
######################


num_workers=8
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []



# --- PRIMA DEL CICLO FOR ---
# Crea i dataset una volta sola per evitare sprechi
dataset_train_obj = Nuc_Dataset(data_train, max_length=201, rc_augmentation=True)
dataset_test_obj = Nuc_Dataset(data_test, max_length=201, rc_augmentation=True) 

# --- DENTRO IL CICLO FOR ---
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n--- Fold {fold+1} ---")

    # USARE GLI OGGETTI DATASET, NON LE LISTE GREZZE
    train_subset_cv = Subset(dataset_train_obj, train_idx) 
    test_subset_cv = Subset(dataset_train_obj, val_idx)
    
    # DataLoader Train
    dataloader_train_cv = DataLoader(
        train_subset_cv, 
        batch_size=32, 
        shuffle=True,
        worker_init_fn=seed_worker,   
        generator=g,
        drop_last=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # DataLoader Validation (Fold test)
    dataloader_test_cv = DataLoader(
        test_subset_cv, 
        batch_size=32,
        shuffle=False,
        worker_init_fn=seed_worker,   
        generator=g,
        drop_last=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )

    # DataLoader External Test (Resting)
    # Nota: Qui usi dataset_test_obj intero, non un subset
    dataloader_test_cv_resting = DataLoader(
        dataset_test_obj, 
        batch_size=32,
        shuffle=False,
        worker_init_fn=seed_worker,   
        generator=g,
        drop_last=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )







# for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#     print(f"\n--- Fold {fold+1} ---")

#     # Divisione fold in train/val
#     train_labels = labels[train_idx]

#     train_subset_cv = Subset(data_train, train_idx)
#     dataloader_train_cv = DataLoader(train_subset_cv, batch_size=32, shuffle=True,worker_init_fn=seed_worker,  
#     generator=g,drop_last=True,num_workers=num_workers,  # <--- AGGIUNGI QUESTO (parallelizza il caricamento)
#     pin_memory=True,persistent_workers=True) # Aggiungi questo se hai abbastanza RAM, velocizza le epoche successive alla prima )

    
#     test_subset_cv = Subset(data_train, val_idx)
#     dataloader_test_cv = DataLoader(test_subset_cv, batch_size=32,shuffle=False,worker_init_fn=seed_worker,  
#     generator=g,drop_last=True,num_workers=num_workers,  # <--- AGGIUNGI QUESTO (parallelizza il caricamento)
#     pin_memory=True ,persistent_workers=True)

#     test_subset_cv_resting = data_test
#     dataloader_test_cv_resting = DataLoader(test_subset_cv_resting, batch_size=32,shuffle=False,worker_init_fn=seed_worker,  
#     generator=g,drop_last=True,num_workers=num_workers,  # <--- AGGIUNGI QUESTO (parallelizza il caricamento)
#     pin_memory=True,persistent_workers=True)
    
    # Modello + Pesi
    transf_parameters_att = {'input_dim': dim_embedding, 
                             'dropout_rate': best_hyperparameters['dropout_rate'], 
                             'num_heads': best_hyperparameters['num_heads']}   

    # Calcola il peso inversamente proporzionale alla frequenza
    class_weights = {label: len(data_train) / count for label, count in label_counts.items()}
    
    # Normalizza (opzionale ma consigliato)
    max_weight = max(class_weights.values())
    class_weights = {label: weight / max_weight for label, weight in class_weights.items()}
    
    # Converte in tensore (ordinando le classi correttamente)
    weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
    
    # Crea il dizionario finale
    labels_list = ['label']
    weight_dict = {label: weights_tensor for label in labels_list}
    
    best_epoch = 'None'
    
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

    # Salvataggio dei pesi migliori
    torch.save(best_state_cpu,f"best_model_weights_99_8_percentile_fold{fold}_03_03_26.pt")

    # === RICARICA ===
    best_model = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    state_dict = torch.load(f"best_model_weights_99_8_percentile_fold{fold}_03_03_26.pt", map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)
    best_model.eval()
    
    fold_results.append({
    'fold': fold + 1,
    'test_metrics': test_classification(best_model, dataloader_test_cv_resting, threshold=0.5)[0],
    'best_epoch': best_epoch
    })
    fold_results[-1].update(classification_metrics(best_true_val, best_val_probs))

for metrics in ["Sensitivity_val",
        "Specificity_val",
        "Accuracy_val",
        "MCC_val",
        "AUC_val"]:
    print(metrics, np.mean([i[metrics] for i in fold_results]))


for metrics in ['Sensitivity (Recall)',
  'Specificity',
  'Accuracy',
  'MCC',
  'AUC',
  'PR AUC',]:
    print(metrics, np.mean([i['test_metrics'][metrics] for i in fold_results]))








