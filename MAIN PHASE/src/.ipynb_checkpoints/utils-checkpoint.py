import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

# Opzionale: Necessario solo se decidi di scommentare le parti grafiche (plot)
import matplotlib.pyplot as plt 
# from sklearn.metrics import roc_curve, precision_recall_curve


# def output_model_from_batch_final(batch, model, device,rc=True):

#     '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''

#     #embedding_tot = batch['embedding'].float().to(device)
#     embedding_tot = batch['sequence']

#     if rc:
#         #embedding_tot_rc = batch['embedding_rev'].float().to(device) 
#         embedding_tot_rc = batch['sequence_rev']
    
#     else:
#         #embedding_tot_rc = batch['embedding'].float().to(device) 
#         embedding_tot_rc = batch['sequence']
        

#     labels = batch['label'].to(device)

#     output, importance = model(embedding_tot)
#     output_rc, importance_rc = model(embedding_tot_rc)

#     return output, output_rc, importance, importance_rc, labels



#####################################################################à

#####################################################################à
def output_model_from_batch_final(batch, model, device, rc=True):
    '''
    Estrae i tensori dal batch e fa il forward pass.
    Ora gestisce input_ids e attention_mask invece delle stringhe.
    '''
    
    # --- Forward Sequence ---
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Passiamo i tensori al nuovo forward del modello
    output, importance = model(input_ids, attention_mask)

    output_rc, importance_rc = None, None

    # --- Reverse Complement Sequence ---
    if rc and 'input_ids_rc' in batch:
        input_ids_rc = batch['input_ids_rc'].to(device)
        attention_mask_rc = batch['attention_mask_rc'].to(device)
        
        # Forward pass per la RC
        output_rc, importance_rc = model(input_ids_rc, attention_mask_rc)
    else:
        # Fallback se RC non c'è: usiamo lo stesso output (o gestisci come zero)
        output_rc, importance_rc = output, importance

    labels = batch['label'].to(device)

    return output, output_rc, importance, importance_rc, labels


#####################################################################à

#####################################################################à


###########################################################
#########################################################
# def training_validation_and_test_loop_classification(
#     model, dataloader_train, dataloader_validation, dataloader_test,
#     epochs=20, lr=0.001, patience=10, weight_decay=0, weigth_dict=None
# ):
#     # Assicurati che il modello sia sul device desiderato
#     device = next(model.parameters()).device

#     criterion = nn.BCEWithLogitsLoss(weight=weigth_dict)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     train_mcc_list, val_mcc_list, test_mcc_list = [], [], []
#     loss_train, loss_val, loss_test = [], [], []

#     best_state_cpu = None  # salviamo lo state_dict su CPU per evitare copie GPU
#     best_val_loss, best_epoch = float('inf'), 0

#     for epoch in range(epochs):
#         # === TRAINING ===
#         model.train()
#         total_loss, batch_count = 0.0, 0
#         all_probs, all_labels = [], []

#         for batch in dataloader_train:
#             optimizer.zero_grad()

#             # output_model_from_batch_final deve restituire tensori sul device corretto
#             output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

#             # assicurati che labels siano sul device
#             labels = labels.float().to(device)

#             # calcolo loss (training)
#             loss = criterion(output, labels) + criterion(output_rc, labels)

#             if torch.isnan(loss) or loss.item() < 1e-8:
#                 # pulizia sicura prima di continuare
#                 del loss, output, output_rc, importance, importance_rc, labels
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 continue

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             batch_count += 1

#             # Prendi probabilità e spostale su CPU come numpy (detach per rimuovere grafo)
#             probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
#             all_probs.extend(probs.tolist())
#             all_labels.extend(labels.detach().cpu().numpy().tolist())

#             # --- Pulizia per evitare leak ---
#             # elimina riferimenti a tensori GPU pesanti
#             del output, output_rc, importance, importance_rc, labels, loss, probs
#             # libera memoria Python/GC
#             gc.collect()
#             # informa l'allocatore CUDA che può liberare memoria inutilizzata
#             torch.cuda.empty_cache()
#             # opzionale: sincronizza per sicurezza in debug
#             # torch.cuda.synchronize()

#         # calcoli training
#         train_loss = total_loss / batch_count if batch_count > 0 else 0.0
#         loss_train.append(train_loss)

#         if len(all_probs) > 0:
#             train_preds = (np.array(all_probs) > 0.5).astype(int)
#             train_mcc = matthews_corrcoef(all_labels, train_preds)
#         else:
#             train_mcc = 0.0
#         train_mcc_list.append(train_mcc)

#         # === VALIDATION ===
#         model.eval()
#         val_total_loss, val_batches = 0.0, 0
#         val_probs, val_labels = [], []

#         with torch.no_grad():
#             for batch in dataloader_validation:
#                 output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
#                 labels = labels.float().to(device)

#                 loss = criterion(output, labels) + criterion(output_rc, labels)

#                 val_total_loss += loss.item()
#                 val_batches += 1

#                 probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
#                 val_probs.extend(probs.tolist())
#                 val_labels.extend(labels.detach().cpu().numpy().tolist())

#                 # pulizia temporanei della validazione
#                 del output, output_rc, importance, importance_rc, labels, loss, probs
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         val_loss = val_total_loss / val_batches if val_batches > 0 else 0.0
#         loss_val.append(val_loss)

#         if len(val_probs) > 0:
#             val_preds = (np.array(val_probs) > 0.5).astype(int)
#             val_mcc = matthews_corrcoef(val_labels, val_preds)
#         else:
#             val_mcc = 0.0
#         val_mcc_list.append(val_mcc)

#         # === TEST ===
#         test_total_loss, test_batches = 0.0, 0
#         test_probs, test_labels = [], []

#         with torch.no_grad():
#             for batch in dataloader_test:
#                 output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
#                 labels = labels.float().to(device)

#                 loss = criterion(output, labels) + criterion(output_rc, labels)

#                 test_total_loss += loss.item()
#                 test_batches += 1

#                 probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
#                 test_probs.extend(probs.tolist())
#                 test_labels.extend(labels.detach().cpu().numpy().tolist())

#                 # pulizia temporanei del test
#                 del output, output_rc, importance, importance_rc, labels, loss, probs
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         test_loss = test_total_loss / test_batches if test_batches > 0 else 0.0
#         loss_test.append(test_loss)

#         if len(test_probs) > 0:
#             test_preds = (np.array(test_probs) > 0.5).astype(int)
#             test_mcc = matthews_corrcoef(test_labels, test_preds)
#         else:
#             test_mcc = 0.0
#         test_mcc_list.append(test_mcc)

#         # === LOGGING ===
#         print(f"\nEpoch {epoch+1}/{epochs}")
#         print(f"Train - Loss: {train_loss:.4f}, MCC: {train_mcc:.4f}")
#         print(f"Val   - Loss: {val_loss:.4f}, MCC: {val_mcc:.4f}")
#         print(f"Test  - Loss: {test_loss:.4f}, MCC: {test_mcc:.4f}")
#         print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

#         # === EARLY STOPPING e salvataggio del best model (solo state_dict su CPU) ===
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_epoch = epoch
#             # salva lo state_dict con i tensori trasferiti su CPU => evita di tenere copie GPU in memoria
#             best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             final_test_probs = test_preds.copy() if len(test_probs) > 0 else []       #QUI HO MESSO LE PROBS INVECE DI PREDS
#             best_val_probs = val_probs.copy()
#             best_true_val = val_labels.copy()

#         if epoch - best_epoch >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

#         # Piccola pulizia di fine-epoca
#         gc.collect()
#         torch.cuda.empty_cache()
#         # torch.cuda.synchronize()

#     # Se vuoi restituire un "modello", restituisci lo state_dict su CPU (più leggero)
#     # se vuoi ricaricarlo in seguito:
#     # model.load_state_dict(best_state_cpu)
#     epoch_best = best_epoch + 1
#     return (
#         train_mcc_list, val_mcc_list, loss_train, loss_val,
#         best_val_loss, best_state_cpu, epoch_best,
#         {'label': val_probs}, {'label': val_labels},
#         val_labels, val_probs, test_labels, final_test_probs, best_val_probs, best_true_val
#     )


#######################################################
###################################################
def training_validation_and_test_loop_classification(
    model, dataloader_train, dataloader_validation, dataloader_test,
    epochs=20, lr=0.001, patience=10, weight_decay=0, weigth_dict=None
):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss(weight=weigth_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mcc_list, val_mcc_list, loss_train, loss_val, loss_test = [], [], [], [], []
    best_state_cpu = None
    best_val_loss, best_epoch = float('inf'), 0
    
    # Variabili per memorizzare i migliori risultati
    best_val_probs, best_true_val, final_test_probs, test_labels = [], [], [], []

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        total_loss, batch_count = 0.0, 0
        all_probs, all_labels = [], []

        for batch in dataloader_train:
            optimizer.zero_grad()

            # Chiama la funzione aggiornata
            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            # Calcolo Loss
            loss = criterion(output, labels) + criterion(output_rc, labels)
            
            # Controllo NaN (Opzionale, rallenta leggermente ma è sicuro)
            if torch.isnan(loss):
                print("Warning: NaN loss detected")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Salva probabilità per metriche (staccando dal grafo)
            with torch.no_grad():
                probs = torch.sigmoid((output + output_rc) / 2).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

            # RIMOSSO: gc.collect() e empty_cache() qui. NON FARLO MAI nel loop dei batch!

        train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        loss_train.append(train_loss)
        
        # Calcolo MCC Train rapido
        train_preds = (np.array(all_probs) > 0.5).astype(int)
        train_mcc = matthews_corrcoef(all_labels, train_preds) if len(all_labels) > 0 else 0.0
        train_mcc_list.append(train_mcc)

        # === VALIDATION ===
        model.eval()
        val_total_loss, val_batches = 0.0, 0
        val_probs, val_labels_epoch = [], []

        with torch.no_grad():
            for batch in dataloader_validation:
                output, output_rc, _, _, labels = output_model_from_batch_final(batch, model, device)
                loss = criterion(output, labels) + criterion(output_rc, labels)
                
                val_total_loss += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid((output + output_rc) / 2).cpu().numpy()
                val_probs.extend(probs)
                val_labels_epoch.extend(labels.cpu().numpy())

        val_loss = val_total_loss / val_batches if val_batches > 0 else 0.0
        loss_val.append(val_loss)
        
        val_preds = (np.array(val_probs) > 0.5).astype(int)
        val_mcc = matthews_corrcoef(val_labels_epoch, val_preds) if len(val_labels_epoch) > 0 else 0.0
        val_mcc_list.append(val_mcc)

        # === TEST (Monitoraggio) ===
        # Nota: Se il test set è grande, farlo a ogni epoca rallenta. Valuta se farlo solo alla fine.
        test_total_loss, test_batches = 0.0, 0
        test_probs_epoch, test_labels_epoch = [], []

        with torch.no_grad():
            for batch in dataloader_test:
                output, output_rc, _, _, labels = output_model_from_batch_final(batch, model, device)
                loss = criterion(output, labels) + criterion(output_rc, labels)
                test_total_loss += loss.item()
                test_batches += 1
                
                probs = torch.sigmoid((output + output_rc) / 2).cpu().numpy()
                test_probs_epoch.extend(probs)
                test_labels_epoch.extend(labels.cpu().numpy())

        test_loss = test_total_loss / test_batches if test_batches > 0 else 0.0
        loss_test.append(test_loss)
        
        test_preds = (np.array(test_probs_epoch) > 0.5).astype(int)
        test_mcc = matthews_corrcoef(test_labels_epoch, test_preds) if len(test_labels_epoch) > 0 else 0.0

        # === LOGGING ===
        print(f"Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss:.4f} MCC: {train_mcc:.3f} | Val Loss: {val_loss:.4f} MCC: {val_mcc:.3f} | Test Loss: {test_loss:.4f}")

        # === EARLY STOPPING ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Salvataggio leggero
            best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Salviamo i risultati migliori
            best_val_probs = val_probs
            best_true_val = val_labels_epoch
            final_test_probs = test_probs_epoch # Salva le probs, non le preds (più flessibile)
            test_labels = test_labels_epoch

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Pulizia memoria SOLO a fine epoca
        gc.collect()
        torch.cuda.empty_cache()

    return (
        train_mcc_list, val_mcc_list, loss_train, loss_val,
        best_val_loss, best_state_cpu, best_epoch + 1,
        {'label': best_val_probs}, {'label': best_true_val},
        best_true_val, best_val_probs, test_labels, final_test_probs, best_val_probs, best_true_val
    )



#########################################################################à
#########################################################################à
# def test_classification(model, dataloader_test, threshold=0.5):
#     device = next(model.parameters()).device
#     model.eval()

#     val_labels, val_preds = [], []
#     importance_list = []
#     importance_rc_list = []
#     with torch.no_grad():
#         for batch in dataloader_test:
#             output, output_rc, importance, importance_rc, labels= output_model_from_batch_final(batch, model, device)

#             probs = torch.sigmoid((output+output_rc)/2).cpu().numpy()

#             val_preds.extend(probs)
#             val_labels.extend(labels.cpu().numpy())
#             importance_list.append(importance)
#             importance_rc_list.append(importance_rc)

#     metrics = classification_metrics(val_labels, val_preds, threshold=threshold)
#     return metrics, val_labels, val_preds, importance_list, importance_rc_list,val_preds
############################################################################################################à
##################################################################################################
def test_classification(model, dataloader_test, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()

    val_labels, val_probs = [], []
    importance_list, importance_rc_list = [], []
    
    with torch.no_grad():
        for batch in dataloader_test:
            # Usa la funzione helper aggiornata
            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            probs = torch.sigmoid((output + output_rc)/2).cpu().numpy()

            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy())
            
            # Salvataggio importance (attenzione se sono tensori pesanti, meglio portarli su CPU)
            if importance is not None:
                importance_list.append(importance.cpu())
            if importance_rc is not None:
                importance_rc_list.append(importance_rc.cpu())

    # Calcolo metriche
    metrics = classification_metrics(val_labels, val_probs, threshold=threshold)
    
    return metrics, val_labels, val_probs, importance_list, importance_rc_list, val_probs


def classification_metrics(y_true, y_pred_probs, threshold=0.5):

    y_true = np.array(y_true).ravel()
    y_pred_probs = np.array(y_pred_probs).ravel()
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('final threshold', threshold)

    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)

    save_dir = "../data"
    os.makedirs(save_dir, exist_ok=True)


    df_results = pd.DataFrame({
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,  
        'y_pred': y_pred               
    })
    
    csv_path = os.path.join(save_dir, "predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"File CSV con predizioni salvato in {csv_path}")

    # # ==================== ROC CURVE ====================
    # fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    # plt.figure(figsize=(6, 6))
    # plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    # plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate (Sensitivity)")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.tight_layout()

    # roc_path = os.path.join(save_dir, "roc_AUC.png")
    # plt.savefig(roc_path, dpi=300)
    # plt.close()
    # print(f"ROC curve salvata in {roc_path}")

    # # ==================== PR CURVE ====================
    # precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)

    # plt.figure(figsize=(6, 6))
    # plt.plot(recall, precision, linewidth=2)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title(f"Precision-Recall Curve (AP = {pr_auc:.3f})")
    # plt.tight_layout()

    # pr_path = os.path.join(save_dir, "pr_curve.png")
    # plt.savefig(pr_path, dpi=300)
    # plt.close()
    # print(f"PR curve salvata in {pr_path}")

    # Output metriche
    output = {
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "MCC": mcc,
        "AUC": auc,
        "PR AUC": pr_auc
    }

    print(output)
    return output









