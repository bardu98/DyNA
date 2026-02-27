import os
import gc                       # Per il Garbage Collector (gc.collect)
import torch
import torch.nn as nn           # Per nn.BCEWithLogitsLoss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Per i grafici (plt.figure, plt.plot, ecc.)

# Tutte le metriche di scikit-learn utilizzate
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)



def output_model_from_batch_final(batch, model, device):
    embedding_tot = batch['embedding'].float().to(device)
    labels = batch['label'].to(device)
    
    output, importance = model(embedding_tot)

    embedding_tot_rc = batch['embedding_rev'].float().to(device) 
    output_rc, importance_rc = model(embedding_tot_rc)

    return output, output_rc, importance, importance_rc, labels



def training_validation_and_test_loop_classification(
    model, dataloader_train, dataloader_validation, dataloader_test,
    epochs=20, lr=0.001, patience=10, weight_decay=0, weigth_dict=None
):
    device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss(weight=weigth_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mcc_list, val_mcc_list, test_mcc_list = [], [], []
    loss_train, loss_val, loss_test = [], [], []

    best_state_cpu = None
    best_val_loss, best_epoch = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        total_loss, batch_count = 0.0, 0
        all_probs, all_labels = [], []

        for batch in dataloader_train:
            optimizer.zero_grad()

            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            labels = labels.float().to(device)

            loss = criterion(output, labels) + criterion(output_rc, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Prendi probabilità e spostale su CPU come numpy (detach per rimuovere grafo)
            probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())


        # calcoli training
        train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        loss_train.append(train_loss)

        if len(all_probs) > 0:
            train_preds = (np.array(all_probs) > 0.5).astype(int)
            train_mcc = matthews_corrcoef(all_labels, train_preds)
        else:
            train_mcc = 0.0
        train_mcc_list.append(train_mcc)

        # === VALIDATION ===
        model.eval()
        val_total_loss, val_batches = 0.0, 0
        val_probs, val_labels = [], []

        with torch.no_grad():
            for batch in dataloader_validation:
                output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
                labels = labels.float().to(device)

                loss = criterion(output, labels) + criterion(output_rc, labels)

                val_total_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())


        val_loss = val_total_loss / val_batches if val_batches > 0 else 0.0
        loss_val.append(val_loss)

        if len(val_probs) > 0:
            val_preds = (np.array(val_probs) > 0.5).astype(int)
            val_mcc = matthews_corrcoef(val_labels, val_preds)
        else:
            val_mcc = 0.0
        val_mcc_list.append(val_mcc)

        test_total_loss, test_batches = 0.0, 0
        test_probs, test_labels = [], []

        with torch.no_grad():
            for batch in dataloader_test:
                output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
                labels = labels.float().to(device)

                loss = criterion(output, labels) + criterion(output_rc, labels)

                test_total_loss += loss.item()
                test_batches += 1

                probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
                test_probs.extend(probs.tolist())
                test_labels.extend(labels.detach().cpu().numpy().tolist())


        test_loss = test_total_loss / test_batches if test_batches > 0 else 0.0
        loss_test.append(test_loss)

        if len(test_probs) > 0:
            test_preds = (np.array(test_probs) > 0.5).astype(int)
            test_mcc = matthews_corrcoef(test_labels, test_preds)
        else:
            test_mcc = 0.0
        test_mcc_list.append(test_mcc)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, MCC: {train_mcc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MCC: {val_mcc:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, MCC: {test_mcc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            final_test_probs = test_probs.copy() if len(test_probs) > 0 else []
            best_val_probs = val_probs.copy()
            best_true_val = val_labels.copy()

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    epoch_best = best_epoch + 1
    return (
        train_mcc_list, val_mcc_list, loss_train, loss_val,
        best_val_loss, best_state_cpu, epoch_best,
        {'label': val_probs}, {'label': val_labels},
        val_labels, val_probs, test_labels, final_test_probs, best_val_probs, best_true_val
    )




def test_classification(model, dataloader_test, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()

    val_labels, val_preds = [], []
    importance_list = []
    importance_rc_list = []
    with torch.no_grad():
        for batch in dataloader_test:
            output, output_rc, importance, importance_rc, labels= output_model_from_batch_final(batch, model, device)

            probs = torch.sigmoid((output+output_rc)/2).cpu().numpy()

            val_preds.extend(probs)
            val_labels.extend(labels.cpu().numpy())
            importance_list.append(importance)
            importance_rc_list.append(importance_rc)

    metrics = classification_metrics(val_labels, val_preds, threshold=threshold)
    return metrics, val_labels, val_preds, importance_list, importance_rc_list,val_preds



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
