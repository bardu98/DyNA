# DyNA (CadmusDNA): Interpretable Nucleosome Prediction

This repository contains the official implementation of the **DyNA** (also referred to as **CadmusDNA** in the codebase) architecture, a Transformer-based model designed for high-accuracy and interpretable nucleosome positioning prediction. 

The pipeline is divided into two main phases: an initial benchmarking/hyperparameter optimization phase, and the main training/testing phase which includes downstream explainability studies.

---

## 🛠️ Phase 1: Benchmarking & Hyperparameter Tuning

This initial phase uses the external *Homo sapiens* (HS) dataset to establish baselines and find the optimal hyperparameters for the model.

### 1. Data Processing
To process the external HS data and generate the necessary embeddings, use:
`data_processing_embedding.py`

### 2. Hyperparameter Optimization
We utilized the Optuna framework on the external HS dataset to search for the best model configuration. To run the hyperparameter search, use:
`Cadmus_hyperparameters.py`

---

## 🚀 Phase 2: Main Phase (Training & Testing)

In the main experimental phase, the model is trained on **Lymphoblastoid Human Cells** and tested on **CD4+ T Human Cells** to evaluate cross-cell-line generalization.

### 1. Data Preprocessing Pipeline
The raw MNase-seq data is provided as WIG files. Follow these steps in order to process the raw data into the final format required by the model:

* **Step 1.1: Genome Assembly Conversion (hg18 -> hg19)**
  The original Train and Test WIG files are mapped to the `hg18` assembly. Convert the mapping to `hg19` using:
  `Convert_Genome_hg18_to_hg19.py`

* **Step 1.2: WIG to CSV Conversion**
  Extract the relevant sequence coordinates and labels from the hg19 WIG files by running:
  `creation_data_from_wig_to_csv.ipynb`

* **Step 1.3: CSV to PKL Conversion**
  Convert the CSV files into serialized Pickle (`.pkl`) objects for efficient dataloading during training:
  `creation_data_from_csv_to_pkl.ipynb`

### 2. Model Training
To train the CadmusDNA model and obtain the final weights, run the main training script. This script utilizes **5-fold cross-validation**:
`train_CadmusDNA.py`

*Output:* This will generate 5 distinct model weight files corresponding to each fold, saved as: 
`best_model_weights_99_8_percentile_fold{fold}.pt` (where `{fold}` ranges from 0 to 4).

---

## 🔍 Phase 3: Explainability

One of the core strengths of CadmusDNA is its ability to extract biological insights directly from the model's self-attention mechanisms.

**⚠️ Prerequisite:** Before analyzing explainability, you must run the predictions notebook to produce the prediction files using the trained `.pt` weights.

Once the predictions are generated, you can discover interesting mechanistic and regulatory features by running the explainability notebook:
`CadmusDNA_explainability_study.ipynb`

---

## 🚀 Inference

To generate predictions and extract attention matrices on new data, you can use the `inference.py` script via the command line.

### 📦 Required Data Format
The input dataset must be a `.pkl` (Pickle) file containing a **list of dictionaries**. Each dictionary represents a single sample and **must** contain the keys `'sequence'` (the forward DNA sequence) and `'rev_sequence'` (its reverse complement). 

*Example of the `.pkl` structure:*
```python
[
    {
        'sequence': 'GATGAGTAGAATCCCCCAGAAAGGAG...',
        'rev_sequence': '...CTCCTTTCTGGGGGATTCTACTCATC'
    },
    {
        'sequence': 'ATGCGTACGTAGCTAGCTAGCTAGCA...',
        'rev_sequence': '...TGCTAGCTAGCTAGCTACGTACGCAT'
    }
]
```

### 💻 How to Run the Script

Execute the script from the terminal by specifying the path to your dataset and the trained model weights (`.pt`):

```bash
python inference.py \
  --dataset "path/to/your/dataset.pkl" \
  --weights "path/to/your/best_model_weights.pt" \
  --output_dir "./results" \
  --batch_size 32
```

### ⚙️ Available Parameters

| Argument | Description | Default | Required |
| :--- | :--- | :---: | :---: |
| `--dataset` | Path to the `.pkl` file containing the data to analyze. | - | ✅ Yes |
| `--weights` | Path to the model weights file (`.pt`). | - | ✅ Yes |
| `--output_dir`| Directory where results will be saved. Created automatically if it doesn't exist. | `./results` | ❌ No |
| `--batch_size`| Batch size for inference (reduce if you encounter VRAM/memory issues). | `32` | ❌ No |

### 📂 Generated Output Files

Upon completion, you will find the following files in the directory specified by `--output_dir` (e.g., `./results`):
1. 📄 **`predictions.csv`**: A CSV file containing the computed probabilities (`probabilities`) and the predicted labels (`predictions`) for each sequence.
2. 🧠 **`attention_matrices.pkl`**: A Pickle file containing the raw attention matrix tensors for each sample, ready for downstream analysis.
3. 📊 **`attention_sample_0.png`**: A bar chart (PNG) visually displaying the Attention Score across 6-mers for the first sample in your dataset.
