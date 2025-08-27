# Banking77 Dataset Overview

## Description
The Banking77 dataset is a benchmark for intent classification in the banking domain, consisting of online customer queries annotated with their corresponding intents. It was originally introduced in the paper "Efficient Intent Detection with Dual Sentence Encoders" by Casanueva et al. (2020). The dataset focuses on real-world banking scenarios, such as card issues, transfers, and account inquiries, making it highly relevant for developing conversational AI systems like chatbots in financial services.

- **Source**: Available on Hugging Face as `mteb/banking77` (part of the Massive Text Embedding Benchmark - MTEB).
- **Task Type**: Text classification (intent detection).
- **Domains**: Written text from banking queries.
- **Number of Intents**: 77 unique intents (e.g., "card_arrival", "card_declined", "transfer_issues").
- **Sample Structure**:
  - `text`: The customer query (e.g., "I am still waiting on my card?").
  - `label`: Integer label (e.g., 11 for "card_arrival").
  - `label_text`: Human-readable intent name (e.g., "card_arrival").
- **Example Queries**:
  - "What can I do if my card still hasn't arrived after 2 weeks?" → Label: card_arrival
  - "I have been waiting over a week. Is the card still coming?" → Label: card_arrival

The dataset is designed for evaluating embedding models and classifiers on intent detection, with a focus on domain-specific NLP challenges.

## Dataset Statistics
Based on the provided splits:
- **Train Split**:
  - Samples: 10,003
  - Average Text Length: ~59 characters
  - Min/Max Text Length: 13 / 433 characters
  - Unique Labels: 77
  - Label Distribution: Imbalanced (e.g., some intents have 182 samples, others as low as 35)
- **Test Split**:
  - Samples: 3,080
  - Average Text Length: ~54 characters
  - Min/Max Text Length: 13 / 368 characters
  - Unique Labels: 77
  - Label Distribution: Balanced (exactly 40 samples per intent)
- **Total Characters**: Train (~594,916), Test (~167,036)
- **Unique Texts**: 100% unique in each split; no overlap between train and test.

The imbalance in the train split reflects real-world scenarios where certain intents (e.g., common queries like card arrivals) occur more frequently, adding realism to model training.

## Why I Chose This Dataset
I selected the Banking77 dataset for this project because it aligns perfectly with the goal of building a production-ready intent classification system for banking applications. It provides a realistic, domain-specific challenge that goes beyond general-purpose NLP datasets like GLUE or SNLI.

- **Relevance to Project**: The project guide focuses on fine-tuning BERT variants (e.g., LoRA, DoRA, QLoRA) and advanced techniques like Mixture-of-Experts (MoE) for intent classification. Banking77's banking-focused queries enable direct application to real-world use cases, such as chatbots for customer support automation or fraud detection.
- **Benchmark Value**: As part of MTEB, it allows easy evaluation and comparison with state-of-the-art embedding models. This ensures our experiments can be benchmarked against published results, facilitating reproducibility and progress tracking.
- **Practicality**: The dataset size (13,083 total samples) is manageable for intern-level projects on Google Colab, yet large enough to demonstrate fine-tuning effectiveness without requiring massive compute resources.

## Novel Value of This Dataset
Banking77 introduces novel value in NLP research and applications by addressing underrepresented aspects of intent detection:
- **Domain Specificity**: Unlike generic datasets (e.g., ATIS for airlines), it targets the banking sector, which involves sensitive, regulated language (e.g., queries about PINs, transfers, or declined cards). This enables exploration of domain adaptation, where models must handle financial jargon and contextual nuances.
- **Multi-Intent Granularity**: With 77 fine-grained intents, it challenges models to distinguish subtle differences (e.g., "card_arrival" vs. "card_lost"). This is more complex than coarser datasets, highlighting the need for advanced techniques like parameter-efficient fine-tuning to avoid overfitting on rare intents.
- **Real-World Imbalance and Noise**: The train split's label imbalance mirrors production environments, encouraging techniques like data augmentation or weighted loss functions. Its queries are derived from actual user interactions, adding authenticity and testing robustness to variations in phrasing.
- **Conversational AI Focus**: It supports building efficient dual-sentence encoders for intent detection, as per the original paper, which is novel for low-resource settings. In our project, this novelty extends to comparing PEFT methods (e.g., QLoRA for memory efficiency) on a dataset that demands high accuracy for user-facing systems.
- **Interdisciplinary Impact**: Banking77 bridges NLP with fintech, enabling innovations in automated customer service, where misclassifying intents could lead to user frustration or compliance issues. Its inclusion in MTEB further amplifies its value for multilingual and embedding-based benchmarks.

By using Banking77, this project not only replicates standard benchmarks but also contributes novel insights into efficient fine-tuning for domain-specific, imbalanced classification tasks.

## Additional Pointers
- **How to Load and Use**:
  - Via Hugging Face: `from datasets import load_dataset; dataset = load_dataset("mteb/banking77")`
  - Evaluation: Use MTEB for embedding models: `import mteb; task = mteb.get_task("Banking77Classification"); evaluator = mteb.MTEB(task); evaluator.run(your_model)`
  - Metrics: Typically accuracy and macro-F1 due to multi-class nature; focus on handling imbalance.
- **Challenges**:
  - Label Imbalance: Train has varying counts (35–182 per intent), requiring strategies like oversampling.
  - Text Variability: Queries range from short ("Where is my card?") to detailed, testing contextual understanding.
- **Benefits for Project**:
  - Enables daily experiments (e.g., Day 2: Full BERT fine-tuning) with quick iterations.
  - Supports wandb logging for hyperparameters, metrics, and comparisons across methods.
- **Citations**:
  - Original Paper: Casanueva et al. (2020) - Efficient Intent Detection with Dual Sentence Encoders.
  - MTEB: Muennighoff et al. (2022) and Enevoldsen et al. (2025).
- **Related Resources**:
  - GitHub: MTEB Repository for more benchmarks.
  - Applications: Ideal for chatbots, virtual assistants in banking (e.g., integrating with Rasa or Dialogflow).
  - Extensions: Combine with data augmentation (e.g., paraphrasing) for future work, as suggested in the project guide.

This dataset provides a strong foundation for exploring advanced NLP techniques while delivering practical value in the fintech space.