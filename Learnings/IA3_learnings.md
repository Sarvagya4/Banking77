Fine-Tuning the Banking77 Dataset with IA3 PEFT: Key Learnings
Introduction to IA3 PEFT
IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) is a Parameter-Efficient Fine-Tuning (PEFT) method designed to adapt large pre-trained language models to downstream tasks with minimal additional parameters. Introduced in the paper "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" by Liu et al. (2022), IA3 focuses on injecting lightweight adapter vectors that scale and shift the inner activations of the model, specifically targeting key-value projections in attention layers and feed-forward layers.
Unlike full fine-tuning, which updates all model parameters and can be computationally expensive, IA3 only introduces a small number of trainable parameters (typically less than 0.1% of the total), making it highly efficient for resource-constrained environments. This approach preserves the original model's knowledge while enabling task-specific adaptations, reducing the risk of catastrophic forgetting.
Why I Picked IA3 for Training
I chose IA3 for fine-tuning the Banking77 dataset because it strikes an excellent balance between performance and efficiency. The Banking77 dataset, which consists of 10,003 training samples across 77 intent classes for banking-related queries, requires a model that can handle multi-class classification without overfitting or requiring excessive computational resources.
Key reasons for selecting IA3:

Parameter Efficiency: IA3 adds very few parameters (e.g., scaling vectors for query, key, and value in transformers), allowing fine-tuning on a single GPU without needing massive hardware.
Preservation of Pre-trained Knowledge: It minimizes disruption to the base model's weights, which is crucial for domain-specific tasks like banking intents where general language understanding from pre-training is valuable.
Faster Training and Inference: Compared to methods like LoRA or full fine-tuning, IA3 has shown competitive results with lower training time, making it ideal for iterative experiments with hyperparameter tuning via Optuna.
Compatibility with Hugging Face PEFT Library: Easy integration with transformers like RoBERTa, enabling quick setup for sequence classification.

In my experiments, IA3 achieved a test accuracy of ~72.5% and F1-score of ~70.8% on the Banking77 test set after hyperparameter optimization, demonstrating its effectiveness.
Real-Life Example: Understanding IA3 for Long-Term Memory
Imagine a seasoned librarian (the pre-trained model) who has memorized the layout of an entire library (general knowledge). Now, you want her to specialize in recommending books for banking and finance queries without forgetting the rest of the library.

Without IA3 (Full Fine-Tuning): You'd retrain her entirely on banking books, risking her forgetting other sections (catastrophic forgetting) and taking a lot of time/energy.
With IA3: You give her a simple "cheat sheet" (adapter vectors) that amplifies (scales up) relevant banking-related memories and inhibits (scales down) irrelevant ones when a query comes in. This "cheat sheet" is lightweight and doesn't overwrite her core knowledge.

For long-term memory: IA3 acts like synaptic scaling in the brain, where neurons adjust connection strengths (via multiplication factors) to retain old memories while forming new ones. In the librarian analogy, the cheat sheet becomes part of her long-term routine, allowing her to recall banking specifics indefinitely without relearning everything.
This makes IA3 suitable for applications like chatbots in banking apps, where the model needs to retain broad language skills while specializing in intents like "card_payment_fee_charged" or "transfer_not_received".
Architecture of IA3
IA3 modifies the transformer architecture by inserting learnable scaling vectors into specific components:

In Attention Layers: Multiplies the key (K) and value (V) vectors by learnable vectors l_k and l_v to amplify/inhibit activations.
In Feed-Forward Layers: Scales the intermediate activations with l_ff.
Formula: For a layer's output, it's adjusted as output = input * l where l is the IA3 vector (initialized near 1 for minimal initial change).

This results in efficient adaptation without adding new layers or matrices.
Mermaid Diagram
graph TD
    A[Input Embeddings] --> B[Transformer Layer]
    subgraph Transformer Layer
        B --> C[Multi-Head Attention]
        C -->|Query Q| D[No Change]
        C -->|Key K * l_k| E[Scaled Key]
        C -->|Value V * l_v| F[Scaled Value]
        E --> G[Attention Computation]
        F --> G
        D --> G
        G --> H[Feed-Forward Network]
        H -->|Intermediate * l_ff| I[Scaled FFN]
        I --> J[Layer Output]
    end
    J --> K[Next Layer or Classifier]
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px

This diagram illustrates how IA3 injects scaling vectors (l_k, l_v, l_ff) into the attention and FFN components.
Code Snippet
Here's a snippet from the Jupyter notebook showing how to set up IA3 with Hugging Face's PEFT library for fine-tuning RoBERTa on Banking77:
from peft import IA3Config, get_peft_model
from transformers import AutoModelForSequenceClassification

# Load base model
model_name = "roberta-base"
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Configure IA3
ia3_config = IA3Config(
    task_type="SEQ_CLS",  # Sequence classification
    target_modules=["query", "value"],  # Scale query and value in attention
    feedforward_modules=[]  # Optional: Can include FFN if needed
)

# Get PEFT model
model = get_peft_model(base_model, ia3_config)

# Print trainable parameters (shows efficiency)
model.print_trainable_parameters()

This code initializes IA3 adapters, resulting in only ~0.01% trainable parameters compared to the full model.
Challenges Faced in Training
Initially, I attempted to train IA3 on BERT-base-uncased, but encountered several issues:

Convergence Problems: BERT struggled with slow convergence on the Banking77 dataset, achieving only ~60% validation accuracy after several epochs, likely due to its less robust pre-training on diverse texts compared to RoBERTa.
Overfitting Early: With BERT, the model overfit quickly on the imbalanced intents, leading to poor generalization (F1-score dropping after epoch 3).
Memory and Speed: BERT's architecture caused higher VRAM usage during fine-tuning, making hyperparameter searches with Optuna slower.

Switching to RoBERTa-base resolved these:

RoBERTa's dynamic masking and larger pre-training corpus improved initial representations for banking texts.
It converged faster, reaching ~73% validation accuracy.
Better handling of long sequences in queries, reducing padding-related inefficiencies.

Overall, this highlighted the importance of base model selection for PEFT methods like IA3.