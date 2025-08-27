# Model Training Results (Day 3)

## What I Have Done
I fine-tuned a BERT model on the Banking77 dataset for intent classification using LoRA (Low-Rank Adaptation) technique. 
Two different training experiments were conducted with varying epochs and hyperparameters to assess performance.

## Why I Did This
The goal was to compare LoRA fine-tuning with standard full fine-tuning of BERT, focusing on efficiency in terms of training time, parameter updates, and model performance metrics like accuracy and F1 score. This helps determine if LoRA can achieve comparable or better results with reduced computational resources.

## What I Observed
- **Training 1 (LoRA)**: Over 5 epochs, the model demonstrated rapid improvement. Accuracy rose from ~75% to ~88%, and the F1 score from ~0.74 to ~0.88, indicating strong convergence with minimal overfitting.
- **Training 2 (LoRA)**: With 6 epochs, performance further enhanced, reaching ~90% accuracy and ~0.90 F1 score. The extended training allowed for better generalization, with validation loss stabilizing around 0.39.

### Observations in LoRA
- LoRA fine-tuning showed faster convergence compared to standard fine-tuning, with significant improvements in early epochs.
- The model achieved higher final metrics (e.g., accuracy up to 89.7% in Training 2) while updating far fewer parameters, making it more parameter-efficient.
- Validation loss decreased steadily, suggesting LoRA helps mitigate overfitting by focusing adaptations on low-rank matrices rather than the entire model.
- Overall, LoRA maintained high performance with reduced risk of catastrophic forgetting of pre-trained knowledge.

### Difference Between Normal Fine-Tuning and LoRA
- **Normal Fine-Tuning**: Involves updating all parameters of the model, which can be computationally expensive and memory-intensive. It risks overwriting pre-trained weights, potentially leading to overfitting or loss of generalization. From previous experiments (Day 2), it achieved up to ~84% accuracy over 6 epochs.
- **LoRA Fine-Tuning**: Only trains low-rank decomposition matrices added to the model's weights, freezing the original parameters. This reduces the number of trainable parameters (often by 99% or more), making it faster, more memory-efficient, and easier to deploy multiple adapters. It preserves the base model's knowledge better and allows for modular adaptations.

### Time Difference
- **Normal Fine-Tuning (Day 2)**: Training 1 (3 epochs) and Training 2 (6 epochs) times were not explicitly recorded, but standard BERT fine-tuning typically takes longer due to full parameter updates. Based on common benchmarks, it can take 1-2x longer per epoch on similar hardware.
- **LoRA Fine-Tuning (Day 3)**: Training 1 completed in ~31 minutes for 5 epochs, and Training 2 in ~81 minutes for 6 epochs. LoRA is generally 2-3x faster per epoch because fewer parameters are updated, leading to quicker training overall.

### Accuracy Difference
- **Normal Fine-Tuning (Day 2)**: Peak accuracy was ~82.9% (Training 1, 3 epochs) and ~84.4% (Training 2, 6 epochs).
- **LoRA Fine-Tuning (Day 3)**: Achieved higher peaks at ~88.1% (Training 1, 5 epochs) and ~89.7% (Training 2, 6 epochs). LoRA outperformed normal fine-tuning by 4-5% in accuracy, likely due to efficient adaptation without full retraining.

## Results

### Training 1 (LoRA)
| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|-------|---------------|-----------------|----------|----|
| 1 | 2.690700 | 1.064127 | 0.752248 | 0.736822 |
| 2 | 0.900400 | 0.563461 | 0.851149 | 0.846966 |
| 3 | 0.530900 | 0.450335 | 0.866134 | 0.864941 |
| 4 | 0.396200 | 0.394920 | 0.880120 | 0.879420 |
| 5 | 0.312700 | 0.385986 | 0.881119 | 0.880587 |

## Training 1 Metrics (LoRA)
![Training 1 Metrics](images\LoRA_T_!.png)

### Training 2 (LoRA)
| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|-------|---------------|-----------------|----------|----|
| 1 | 3.046400 | 1.453697 | 0.660340 | 0.631380 |
| 2 | 1.184700 | 0.745732 | 0.819181 | 0.811022 |
| 3 | 0.621700 | 0.494968 | 0.880120 | 0.877288 |
| 4 | 0.395500 | 0.427494 | 0.892108 | 0.891564 |
| 5 | 0.268600 | 0.408421 | 0.895105 | 0.893964 |
| 6 | 0.211000 | 0.394300 | 0.897103 | 0.896149 |

## Training 2 Metrics (LoRA)
![Training 2 Metrics](images\LoRA_T_2.png)