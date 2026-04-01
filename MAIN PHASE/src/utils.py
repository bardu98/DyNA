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

import matplotlib.pyplot as plt 


def output_model_from_batch_final(batch, model, device, rc=True):
    '''
    Estrae i tensori dal batch e fa il forward pass.
    Ora gestisce input_ids e attention_mask invece delle stringhe.
    '''
    
    # --- Forward Sequence ---
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    output, importance = model(input_ids, attention_mask)

    output_rc, importance_rc = None, None

    # --- Reverse Complement Sequence ---
    if rc and 'input_ids_rc' in batch:
        input_ids_rc = batch['input_ids_rc'].to(device)
        attention_mask_rc = batch['attention_mask_rc'].to(device)
        
        # Forward pass  RC
        output_rc, importance_rc = model(input_ids_rc, attention_mask_rc)
    else:
        # Fallback se RC non c'è: usiamo lo stesso output (o gestisci come zero)
        output_rc, importance_rc = output, importance

    labels = batch['label'].to(device)

    return output, output_rc, importance, importance_rc, labels


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
    
    best_val_probs, best_true_val, final_test_probs, test_labels = [], [], [], []

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        total_loss, batch_count = 0.0, 0
        all_probs, all_labels = [], []

        for batch in dataloader_train:
            optimizer.zero_grad()

            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            loss = criterion(output, labels) + criterion(output_rc, labels)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            with torch.no_grad():
                probs = torch.sigmoid((output + output_rc) / 2).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())


        train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        loss_train.append(train_loss)
        
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

        # === TEST (monitoring) ===
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

        print(f"Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss:.4f} MCC: {train_mcc:.3f} | Val Loss: {val_loss:.4f} MCC: {val_mcc:.3f} | Test Loss: {test_loss:.4f} Test MCC: {test_mcc:.3f}")

        

        # === EARLY STOPPING ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            best_val_probs = val_probs
            best_true_val = val_labels_epoch
            final_test_probs = test_probs_epoch 
            test_labels = test_labels_epoch

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        gc.collect()
        torch.cuda.empty_cache()

    return (
        train_mcc_list, val_mcc_list, loss_train, loss_val,
        best_val_loss, best_state_cpu, best_epoch + 1,
        {'label': best_val_probs}, {'label': best_true_val},
        best_true_val, best_val_probs, test_labels, final_test_probs, best_val_probs, best_true_val
    )


##################################################################################################

def test_classification(model, dataloader_test, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()

    val_labels, val_probs = [], []
    importance_list, importance_rc_list = [], []
    
    with torch.no_grad():
        for batch in dataloader_test:
            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            probs = torch.sigmoid((output + output_rc)/2).cpu().numpy()

            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy())
            
            if importance is not None:
                importance_list.append(importance.cpu())
            if importance_rc is not None:
                importance_rc_list.append(importance_rc.cpu())

    metrics = classification_metrics(val_labels, val_probs, threshold=threshold)
    
    return metrics, val_labels, val_probs, importance_list, importance_rc_list, val_probs


def classification_metrics(y_true, y_pred_probs, threshold=0.5):

    y_true = np.array(y_true).ravel()
    y_pred_probs = np.array(y_pred_probs).ravel()
    y_pred = (y_pred_probs >= threshold).astype(int)

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









