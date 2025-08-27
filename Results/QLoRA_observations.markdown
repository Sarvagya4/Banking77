# Observations from the QLoRA Finetuning

This document summarizes the results and observations from the QLoRA finetuning experiments conducted on the GPT2ForSequenceClassification model using the microsoft/DialoGPT-medium checkpoint. The experiments involved hyperparameter tuning across multiple trials, with the goal of optimizing model performance as measured by accuracy and F1 score. Below are the detailed results and insights from the trials, the best-performing configuration, and the final evaluation on the test set.

## Experimental Setup
- **Model**: GPT2ForSequenceClassification based on microsoft/DialoGPT-medium.
- **Initialization Note**: In all trials, some weights (`score.weight`) were not initialized from the pretrained checkpoint and were newly initialized, indicating the need for task-specific training.
- **Gradient Checkpointing**: The `use_cache=True` setting was incompatible with gradient checkpointing, so it was set to `use_cache=False` during training.

## Trial Results
Five trials were conducted with varying hyperparameters, including learning rate, batch size, number of training epochs, weight decay, and warmup steps. The performance metrics (Training Loss, Validation Loss, Accuracy, and F1 Score) were recorded for each epoch in each trial.

### Trial 0
- **Parameters**:
  - Learning Rate: 1.2454e-05
  - Batch Size: 16
  - Number of Epochs: 3
  - Weight Decay: 0.0987
  - Warmup Steps: 399
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 4.2668        | 4.2282          | 0.0380   | 0.0254   |
  | 2     | 3.9922        | 3.8808          | 0.1339   | 0.1041   |
  | 3     | 3.7733        | 3.7056          | 0.1799   | 0.1511   |
- **Final Accuracy**: 0.1799
- **Observations**: This trial showed poor performance, with low accuracy and F1 scores, likely due to the low learning rate and limited epochs, which restricted the model's ability to converge effectively.

### Trial 1
- **Parameters**:
  - Learning Rate: 3.1661e-05
  - Batch Size: 16
  - Number of Epochs: 6
  - Weight Decay: 0.0724
  - Warmup Steps: 238
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 3.4718        | 3.1212          | 0.3063   | 0.2715   |
  | 2     | 1.6773        | 1.2909          | 0.7136   | 0.7009   |
  | 3     | 1.0646        | 0.8548          | 0.7921   | 0.7837   |
  | 4     | 0.8387        | 0.6976          | 0.8286   | 0.8255   |
  | 5     | 0.7663        | 0.6338          | 0.8446   | 0.8426   |
  | 6     | 0.6609        | 0.6139          | 0.8456   | 0.8437   |
- **Final Accuracy**: 0.8456
- **Observations**: Significant improvement over Trial 0, with a higher learning rate and more epochs contributing to better convergence. Accuracy and F1 scores improved steadily, reaching a plateau by epoch 6.

### Trial 2
- **Parameters**:
  - Learning Rate: 8.0804e-05
  - Batch Size: 16
  - Number of Epochs: 6
  - Weight Decay: 0.0295
  - Warmup Steps: 182
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 1.3608        | 1.0267          | 0.7446   | 0.7351   |
  | 2     | 0.6918        | 0.5195          | 0.8576   | 0.8558   |
  | 3     | 0.4453        | 0.4241          | 0.8756   | 0.8737   |
  | 4     | 0.3404        | 0.3811          | 0.8896   | 0.8879   |
  | 5     | 0.2807        | 0.3629          | 0.8961   | 0.8952   |
  | 6     | 0.2497        | 0.3537          | 0.9000   | 0.8992   |
- **Final Accuracy**: 0.9000
- **Observations**: Further improvement, with a higher learning rate leading to faster convergence and better performance. The model achieved high accuracy and F1 scores, with minimal overfitting as validation loss remained close to training loss.

### Trial 3
- **Parameters**:
  - Learning Rate: 0.000434
  - Batch Size: 8
  - Number of Epochs: 4
  - Weight Decay: 0.0651
  - Warmup Steps: 219
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 0.6190        | 0.5595          | 0.8461   | 0.8400   |
  | 2     | 0.2965        | 0.3841          | 0.9115   | 0.9114   |
  | 3     | 0.1455        | 0.3567          | 0.9195   | 0.9194   |
  | 4     | 0.0847        | 0.3401          | 0.9225   | 0.9225   |
- **Final Accuracy**: 0.9225
- **Observations**: Switching to a smaller batch size (8) and a higher learning rate improved performance further. The model achieved high accuracy and F1 scores with fewer epochs, suggesting efficient training.

### Trial 4
- **Parameters**:
  - Learning Rate: 0.000462
  - Batch Size: 8
  - Number of Epochs: 4
  - Weight Decay: 0.0905
  - Warmup Steps: 389
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 0.5477        | 0.5826          | 0.8466   | 0.8426   |
  | 2     | 0.2797        | 0.3837          | 0.9110   | 0.9106   |
  | 3     | 0.1456        | 0.3527          | 0.9170   | 0.9170   |
  | 4     | 0.0914        | 0.3300          | 0.9260   | 0.9264   |
- **Final Accuracy**: 0.9260
- **Observations**: This trial achieved the best performance, with the highest accuracy and F1 scores. The combination of a smaller batch size, high learning rate, and moderate weight decay optimized the training process.

## Best Trial
- **Parameters**:
  - Learning Rate: 0.000462
  - Batch Size: 8
  - Number of Epochs: 4
  - Weight Decay: 0.0905
  - Warmup Steps: 389
- **Best Accuracy**: 0.9260
- **Observations**: The best trial (Trial 4) demonstrated that a smaller batch size (8) and a relatively high learning rate (0.000462) with moderate weight decay (0.0905) and warmup steps (389) led to the best balance of training stability and performance. The model achieved a validation accuracy of 0.9260 and an F1 score of 0.9264 after 4 epochs.

## Final Model Training
The final model was trained using the best hyperparameters from Trial 4. The results matched those of Trial 4, confirming the reproducibility of the configuration:
- **Results**:
  | Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
  |-------|---------------|-----------------|----------|----------|
  | 1     | 0.5477        | 0.5826          | 0.8466   | 0.8426   |
  | 2     | 0.2797        | 0.3837          | 0.9110   | 0.9106   |
  | 3     | 0.1456        | 0.3527          | 0.9170   | 0.9170   |
  | 4     | 0.0914        | 0.3300          | 0.9260   | 0.9264   |
- **Training Metrics**:
  - Global Steps: 4004
  - Training Loss: 0.5311
  - Train Runtime: 1568.51 seconds
  - Train Samples per Second: 20.407
  - Train Steps per Second: 2.553
  - Total FLOPs: 7.5417e15
  - Epochs: 4.0

## Test Set Evaluation
The final model was evaluated on the test set, yielding the following results:
- **Test Metrics**:
  - Evaluation Loss: 0.3129
  - Evaluation Accuracy: 0.9351
  - Evaluation F1 Score: 0.9350
  - Evaluation Runtime: 43.5264 seconds
  - Samples per Second: 70.7620
  - Steps per Second: 8.8450
  - Epochs: 4.0
- **Observations**: The model generalized well to the test set, achieving a higher accuracy (0.9351) and F1 score (0.9350) than on the validation set, with a lower evaluation loss (0.3129). This suggests robust performance and minimal overfitting.

## Key Observations
1. **Hyperparameter Impact**:
   - Higher learning rates (e.g., 0.000462 in Trial 4) significantly improved convergence speed and final performance compared to lower rates (e.g., 1.2454e-05 in Trial 0).
   - Smaller batch sizes (8 vs. 16) led to better performance, likely due to increased gradient updates per epoch, allowing finer adjustments to the model weights.
   - Moderate weight decay (0.0905 in Trial 4) helped regularize the model without overly constraining learning.
   - Warmup steps (e.g., 389 in Trial 4) stabilized early training, contributing to better final performance.

2. **Training Dynamics**:
   - Training and validation losses decreased consistently across epochs in all trials, indicating effective learning.
   - Accuracy and F1 scores improved steadily, with diminishing returns after 4–6 epochs, suggesting that 4 epochs were sufficient for optimal performance in the best trials.

3. **Model Generalization**:
   - The final model's test set performance (accuracy: 0.9351, F1: 0.9350) was slightly better than its validation performance, confirming that the model generalized well to unseen data.
   - The low evaluation loss on the test set (0.3129) further supports the model's robustness.

4. **Computational Efficiency**:
   - Trials with smaller batch sizes (e.g., Trial 4) required more steps but achieved better performance with fewer epochs, balancing computational cost and model quality.
   - The final training run took approximately 26 minutes (1568.51 seconds), which is reasonable for the observed performance gains.

## Conclusion
The QLoRA finetuning experiments demonstrated that careful hyperparameter tuning significantly improves the performance of the GPT2ForSequenceClassification model. The best configuration (Trial 4) achieved a validation accuracy of 0.9260 and a test accuracy of 0.9351, with consistent F1 scores. Key factors contributing to success included a high learning rate, smaller batch size, and moderate weight decay. These results highlight the effectiveness of QLoRA for finetuning large language models on downstream tasks, achieving strong performance with reasonable computational resources.