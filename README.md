# Transformer End-to-End Practice

This small project is **only** for learning and experimentation.  It brings together three key components of modern deep-learning workflows:

1. **PyTorch** – the core framework used for implementing and training neural networks.
2. **The Transformer architecture** – an encoder–decoder model that relies on self-attention, introduced in the paper *"Attention Is All You Need"*.
3. **Hugging Face Ecosystem** – `transformers`, `datasets`, and pretrained tokenizers that simplify NLP research and prototyping.

## What’s inside?

* A Jupyter/Colab notebook (`Transformer_end_to_end_practice.ipynb`) that walks through:
  * Implementing a Transformer **from scratch** in PyTorch (no `nn.Transformer` shortcut).
  * Loading an English→Vietnamese translation dataset via `datasets`.
  * Using a pretrained **XLM-Roberta** tokenizer for subword tokenisation.
  * Training the model end-to-end with teacher forcing and validating with greedy decoding.

## Goals of the exercise

* Strengthen understanding of PyTorch tensor operations, `nn.Module` design, and optimisation loops.
* Demystify the Transformer by coding each layer (multi-head attention, feed-forward, positional embeddings, etc.).
* Learn how to plug Hugging Face datasets and tokenisers into a custom model pipeline.

## How to run

1. Install the required libraries (Python ≥ 3.8):
   ```bash
   pip install torch torchvision torchaudio transformers datasets tqdm
   ```
2. Open the notebook and run cell by cell.  A CUDA-capable GPU is recommended but the code will fall back to CPU if necessary.

## Acknowledgements

* Vaswani et al., *"Attention Is All You Need"*.
* Hugging Face for the open-source `transformers` and `datasets` libraries.

---

Feel free to fork, modify, and play with different hyper-parameters, datasets, or decoding strategies – the whole point is to **learn by tinkering**! 🚀
