import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


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
