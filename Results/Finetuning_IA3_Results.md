# Fine-Tuning RoBERTa with IA3 on Banking77 Dataset

## Overview
This notebook demonstrates fine-tuning a RoBERTa-base model using the IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) adapter method for sequence classification on the Banking77 dataset. The process includes:
- Data preprocessing and tokenization.
- Hyperparameter optimization using Optuna (10 trials).
- Final training with the best hyperparameters.
- Evaluation on validation and test sets.

The goal is multi-class classification (77 labels). Metrics focus on accuracy and macro F1-score. Training was performed on a GPU (T4). Weights & Biases (W&B) was used for logging.

Key libraries: Hugging Face Transformers, PEFT (for IA3), Optuna, Datasets, Evaluate.

## Hyperparameter Tuning with Optuna
Optuna performed 10 trials to optimize hyperparameters: `learning_rate`, `batch_size`, `num_train_epochs`, and `weight_decay`. The objective was to maximize validation accuracy.

### Trial Summaries
Each trial's final validation accuracy and parameters are listed below. (Extracted from logged outputs; only key data shown for brevity.)

| Trial | Validation Accuracy | Parameters |
|-------|---------------------|------------|
| 0     | 0.024488            | {'learning_rate': 0.0002333612179474077, 'batch_size': 32, 'num_train_epochs': 4, 'weight_decay': 0.0633089216698163} |
| 1     | 0.739130            | {'learning_rate': 0.00013217439430404193, 'batch_size': 16, 'num_train_epochs': 7, 'weight_decay': 0.010321813929659529} (Best) |
| 2     | 0.042478            | {'learning_rate': 3.190667380159007e-05, 'batch_size': 64, 'num_train_epochs': 7, 'weight_decay': 0.03493862296918063} |
| 3     | 0.704148            | {'learning_rate': 0.0001650211066183207, 'batch_size': 16, 'num_train_epochs': 7, 'weight_decay': 0.010410481123897568} |
| 4     | 0.025987            | {'learning_rate': 1.976207304866414e-05, 'batch_size': 32, 'num_train_epochs': 5, 'weight_decay': 0.09787915241654081} |
| 5     | 0.735132            | {'learning_rate': 0.0001065430799570708, 'batch_size': 16, 'num_train_epochs': 6, 'weight_decay': 0.014353076724619759} |
| 6     | 0.656672            | {'learning_rate': 0.00033235127675082955, 'batch_size': 64, 'num_train_epochs': 6, 'weight_decay': 0.04857177553098746} |
| 7     | 0.047476            | {'learning_rate': 2.4093268046675206e-05, 'batch_size': 16, 'num_train_epochs': 6, 'weight_decay': 0.031623273139924524} |
| 8     | 0.201899            | {'learning_rate': 7.755930950069219e-05, 'batch_size': 16, 'num_train_epochs': 5, 'weight_decay': 0.06134784330032274} |
| 9     | 0.093953            | {'learning_rate': 7.613237902003231e-05, 'batch_size': 16, 'num_train_epochs': 4, 'weight_decay': 0.0422396494741149} |

**Best Trial (Trial 1):**  
- Validation Accuracy: 0.739130  
- Parameters:  
  - Learning Rate: 0.00013217439430404193  
  - Batch Size: 16  
  - Number of Train Epochs: 7  
  - Weight Decay: 0.010321813929659529  

## Final Training with Best Parameters
The model was retrained using the best hyperparameters from Optuna. Training ran for 7 epochs.

### Per-Epoch Metrics
(Extracted from the final trainer output table.)

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation F1 |
|-------|---------------|-----------------|---------------------|---------------|
| 1     | 4.295700      | 4.204616        | 0.044478            | 0.014498      |
| 2     | 4.016000      | 3.641601        | 0.372314            | 0.330674      |
| 3     | 3.108200      | 2.353835        | 0.554723            | 0.517513      |
| 4     | 2.094500      | 1.582701        | 0.650175            | 0.626654      |
| 5     | 1.598500      | 1.273890        | 0.704648            | 0.691441      |
| 6     | 1.381700      | 1.140340        | 0.727636            | 0.716337      |
| 7     | 1.291300      | 1.103214        | 0.734633            | 0.723760      |

- **Total FLOPs:** 1,859,858,425,774,080 (1.86e+15)  
- **Train Runtime:** ~439 seconds (7:19 minutes)  
- **Train Samples per Second:** ~109.3  
- **Train Steps per Second:** ~7.99  

The best model (lowest validation loss) was loaded at the end.

## Final Test Results
Evaluation on the test set (after loading the best checkpoint).

| Metric                  | Value              |
|-------------------------|--------------------|
| Test Loss               | 1.1605722904205322 |
| Test Accuracy           | 0.7253246753246754 |
| Test F1 (Macro)         | 0.7075864702367034 |
| Runtime (seconds)       | 10.192             |
| Samples per Second      | 302.199            |
| Steps per Second        | 18.937             |

## Observations
- Steady improvement in accuracy and F1 across epochs, with diminishing returns after epoch 5.
- Overfitting appears minimal (validation loss decreases consistently).
- Final test accuracy (~72.5%) and F1 (~70.8%) indicate solid performance for a 77-class problem.
- W&B Run: [View here](https://wandb.ai/Banking77/huggingface/runs/bsa03dug) (includes history plots for loss, accuracy, etc.).

## Reproducibility
- Model: `roberta-base`  
- Adapter: IA3 (targeting "query" and "value" modules)  
- Dataset: Banking77 (train/validation/test splits)  
- Seed: Not explicitly set (add for full reproducibility).  
- Environment: Python 3, Colab with T4 GPU.  

For full code and logs, see the notebook.