# Learnings from LoRA Fine-Tuning for Intent Classification

## Introduction

I implemented **Low-Rank Adaptation (LoRA)** to fine-tune a BERT-based model efficiently for intent classification using the **Banking77 dataset**. This document outlines the key learnings from applying LoRA, including its implementation, performance, real-life analogies for better understanding, and the advantages and disadvantages specific to the Banking77 dataset for banking intent classification.

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning (PEFT) method that adapts large language models like BERT with minimal computational cost. Instead of updating all model parameters (e.g., BERT’s 110M parameters), LoRA **freezes the pre-trained weights** and introduces **small, trainable low-rank matrices** to specific layers (e.g., attention or feed-forward). These matrices capture task-specific adaptations, making fine-tuning faster and more memory-efficient.

### Key Parameters:
- **Rank (r)**: Controls the size of low-rank matrices (e.g., `r=8` or `r=16`), determining adaptation capacity.
- **Alpha (α)**: Scaling factor for low-rank updates, balancing their impact.
- **Targeted Layers**: Typically applied to BERT’s attention weights (query, key, value) or feed-forward layers.

---

## Real-Life Examples of LoRA

To make LoRA’s concept memorable, here are relatable analogies:

###  Customizing a Pre-Built House
BERT is like a fully built house with fixed walls (pre-trained weights). LoRA adds **custom decorations or modular furniture** (low-rank matrices) to tailor it for banking intents, saving time and resources compared to rebuilding the house (full fine-tuning).

###  Tuning a Car Engine
Think of BERT as a high-performance car engine. Instead of overhauling the entire engine (full fine-tuning), LoRA adds a **small turbocharger** (low-rank matrices) to boost performance for a specific task, like racing on a banking track.

###  Updating a Cookbook
BERT is a massive cookbook with recipes for all cuisines. LoRA adds **sticky notes with new recipes** (low-rank matrices) for banking-specific intents, keeping the original cookbook intact while adapting it efficiently.

---

## Implementation in the Project

I implemented LoRA as follows:

- **Dataset**: Used the **Banking77 dataset** (10,003 train samples, 2,001 validation samples, 3,080 test samples) with 77 intents, such as `"card_arrival"` (e.g., "I am still waiting on my card?").
- **Model**: Fine-tuned a pre-trained BERT model (likely `BERT-base-uncased`) with a classification head for intent detection.
- **LoRA Setup**:
  - Configured LoRA parameters (e.g., `rank=8`, `alpha=16`) targeting BERT’s attention and feed-forward layers.
  - Used Hugging Face’s `PEFT` library for integration.
- **Training**:
  - Trained on Google Colab with GPU acceleration.
  - Hyperparameters: learning rate ~2e-5, batch size 16 or 32, 3–5 epochs.
  - Logged hyperparameters, training/validation metrics (accuracy, macro-F1), and LoRA configurations to **Weights & Biases (wandb)**.
- **Evaluation**: Compared LoRA’s performance to full fine-tuning, focusing on accuracy, F1 score, and computational efficiency.

---

## Why LoRA Was Used

- **Efficiency**: LoRA updates only a small fraction of parameters (<1% of BERT’s 110M), reducing memory and training time.
- **Scalability**: Suitable for Google Colab’s resource constraints, enabling rapid experimentation.
- **Domain Adaptation**: Adapts BERT to banking-specific intents while minimizing overfitting on rare intents.
- **Comparative Analysis**: Required to evaluate trade-offs against other PEFT methods (e.g., DoRA, AdaLoRA).

---

## Key Learnings

###  Significant Memory Savings
- LoRA reduced GPU memory usage by freezing most parameters.
- **Observation**: ~20–30% lower VRAM than full fine-tuning.

###  Comparable Performance
- Achieved accuracy and macro-F1 within **1–2% of full fine-tuning**.
- Slight drop on rare intents (e.g., 35 samples).

###  Faster Training
- **30–50% faster per epoch** due to fewer trainable parameters.

###  Reduced Overfitting
- Constrained updates helped generalize better on imbalanced data.

###  Integration with wandb
- Clear tracking of rank, alpha, loss, and metrics enabled optimal tuning.

---

## Advantages of LoRA for Banking77

| Advantage | Description |
|--------|-------------|
| **Memory Efficiency** | ~2–3 GB less VRAM, ideal for Colab. |
| **Faster Training** | 30–50% speedup per epoch. |
| **Reduced Overfitting** | Better generalization on rare intents. |
| **Scalability** | Supports deployment on edge devices. |
| **Flexibility** | Easy tuning of `r` and `α` via wandb. |

---

## Disadvantages of LoRA for Banking77

| Disadvantage | Description |
|-----------|-------------|
| **Slight Performance Trade-off** | ~1–2% lower accuracy than full fine-tuning. |
| **Hyperparameter Sensitivity** | Needs tuning of `r` and `α`; suboptimal values hurt performance. |
| **Limited for Complex Intents** | May miss nuances in varied phrasings (e.g., "Where is my card?"). |
| **Implementation Overhead** | Requires PEFT setup and configuration. |

---

## Observations from Results

- **Performance vs. Efficiency**: LoRA achieved ~88–89% accuracy vs. ~90% for full fine-tuning — sufficient for production.
- **Validation Loss**: Stabilized after 3–4 epochs, showing less overfitting.
- **Rare Intents**: Lower F1 scores for low-sample intents (e.g., "maximum how many days get the courier?").
- **Deployment**: Ideal for chatbots on mobile or edge devices.

---

## Architecture

LoRA modifies the weight update process in neural network layers by **decomposing large weight updates into low-rank matrices**.

For a pre-trained weight matrix $ W \in \mathbb{R}^{d \times k} $, instead of updating $ W $ directly, LoRA introduces a **low-rank decomposition**:

$$
\Delta W = B A
$$

where:
- $ B \in \mathbb{R}^{d \times r} $
- $ A \in \mathbb{R}^{r \times k} $
- Rank $ r \ll \min(d, k) $ (e.g., $ r = 8 $)

The updated forward pass becomes:
$$
h = W x + \alpha \cdot B A x = W x + \Delta W_{\text{scaled}} x
$$

This avoids updating $ W $, freezing it instead, and only trains $ A $ and $ B $.

---

## Mathematical Explanation: How LoRA Matrices Work

Let’s break down the math:

1. **Original Linear Layer**:
   $$
   y = W x, \quad W \in \mathbb{R}^{d \times k}, x \in \mathbb{R}^{k}
   $$

2. **LoRA Modification**:
   During fine-tuning, instead of changing $ W $, we add a **low-rank update**:
   $$
   y = W x + \Delta W x
   $$

3. **Low-Rank Decomposition**:
   $$
   \Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
   $$
   - This reduces parameters from $ d \times k $ to $ d \times r + r \times k $.
   - Example: $ d=768, k=768, r=8 $ → $ 768 \times 768 = 589,824 $ vs. $ 768 \times 8 + 8 \times 768 = 12,288 $ → **~98% fewer parameters**.

4. **Scaling**:
   $$
   \Delta W = \frac{\alpha}{r} \cdot B A
   $$
   - $ \alpha $ controls the magnitude of the update.
   - Often, $ \alpha $ is fixed (e.g., 16), and $ r $ is tuned.

This allows LoRA to **approximate full fine-tuning** with **minimal trainable parameters**.

---

## A Snippet of Code (PyTorch Implementation)

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha
        )
        # Freeze original weight
        self.linear.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        out = self.linear(x)
        # Add LoRA update
        lora_update = self.lora(x)
        return out + lora_update

```

## Architecture Flow (Mermaid Diagram)

![](Images\LoRA_architecture.png)



---

---
## Difference Between DoRA and LoRA

| Aspect                  | **LoRA**                                                                 | **DoRA**                                                                 |
|-------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **Core Mechanism**      | Low-rank update: $ \Delta W = B A $, merged as $ W + \alpha B A $        | Decomposes weight: $ W = m \cdot \frac{V}{\|V\|_c} $; LoRA updates $ V $ only |
| **Weight Decomposition**| None; updates applied directly                                           | Explicit decomposition into magnitude and direction                     |
| **Performance**         | Good efficiency, but accuracy gap to FT; sensitive to rank               | Outperforms LoRA (e.g., +3.7% on LLaMA commonsense); robust at low ranks |
| **Parameters Added**    | Minimal (depends on rank)                                                | Slightly more (~0.01% over LoRA) due to magnitude vector $ m $           |
| **Learning Patterns**   | Positive correlation in updates; less expressive                         | Negative slope like FT; better mimics full fine-tuning                   |
| **Inference Overhead**  | None (merges post-training)                                              | None (merges post-training)                                              |
| **Use Cases**           | General PEFT for quick adaptations                                       | High-accuracy needs: VLMs, personalized generation, domain adaptation    |