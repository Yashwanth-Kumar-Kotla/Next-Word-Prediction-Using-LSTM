# Next Word Prediction Using LSTM

A PyTorch implementation of a sequence-to-word language model that predicts the next word in a sentence. Built and trained on two distinct corpora — a custom domain-specific dataset and "The Adventures of Sherlock Holmes" (Project Gutenberg) — to demonstrate generalization across text scales and styles.

---

## Demo

**Custom Corpus (seed: `"artificial"`, 9 auto-complete steps)**
```
artificial intelligence
artificial intelligence is
artificial intelligence is transforming
artificial intelligence is transforming the
artificial intelligence is transforming the world
artificial intelligence is transforming the world in
artificial intelligence is transforming the world in many
artificial intelligence is transforming the world in many different
artificial intelligence is transforming the world in many different ways
```

**Sherlock Holmes Corpus (seed: `"International"`, 9 steps)**
```
International donations
International donations are
International donations are gratefully
International donations are gratefully accepted
International donations are gratefully accepted , but
International donations are gratefully accepted , but we
International donations are gratefully accepted , but we can
International donations are gratefully accepted , but we can not
```

---

## Results

| Corpus | Vocabulary Size | Training Sequences | Epochs | Final Loss | Accuracy |
|---|---|---|---|---|---|
| Custom (40 sentences) | 222 | 302 | 50 | 0.163 | **96.36%** |
| Sherlock Holmes (full novel) | 9,398 | 123,765 | 40 | 0.711 | **86.00%** |

---

## Architecture

```
Input Tokens
     │
     ▼
Embedding Layer  (vocab_size → 100 dims)
     │
     ▼
LSTM Layer       (100 → 150 hidden units, batch_first=True)
     │
     ▼  (final hidden state)
Linear Layer     (150 → vocab_size)
     │
     ▼
Softmax → Predicted Next Word
```

The model processes variable-length padded token sequences and predicts a single next word at each step. During inference, predictions are chained to auto-complete full phrases word-by-word.

---

## How It Works

1. **Tokenization** — Raw text is lowercased and tokenized using NLTK's `word_tokenize`
2. **Vocabulary** — A word-to-index mapping is built from token frequencies via `Counter`
3. **Sequence Construction** — N-gram style: for each sentence, all prefixes are extracted as training samples (`[w1]→w2`, `[w1,w2]→w3`, …)
4. **Padding** — All sequences are left-padded with zeros to match the longest sequence in the corpus
5. **Training** — Adam optimizer + CrossEntropyLoss; CUDA-accelerated when available
6. **Inference** — Input text is numericalized, padded, fed to the model; the argmax over the output logits gives the next word

---

## Tech Stack

- **Framework:** PyTorch
- **NLP Tokenization:** NLTK (`word_tokenize`, `sent_tokenize`)
- **Numerics:** NumPy
- **Hardware:** CUDA GPU (falls back to CPU automatically)
- **Language:** Python 3.12

---

## Project Structure

```
├── Next_Word_Prediction_using_LSTM.ipynb              # Custom corpus notebook
├── Next_Word_Prediction_using_LSTM on kaggle dataset.ipynb  # Sherlock Holmes corpus notebook
└── document.txt                                       # The Adventures of Sherlock Holmes (Project Gutenberg)
```

---

## Quickstart

### Prerequisites

```bash
pip install torch nltk numpy
```

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Run

Open either notebook in Jupyter or Google Colab and run all cells. GPU is recommended for the Sherlock Holmes notebook (123K+ training sequences).

---

## Key Design Decisions

**Why LSTM over a simple RNN?**  
LSTMs use gating mechanisms (input, forget, output gates) to retain long-range dependencies in text — critical for natural language where the relevant context can be many words back.

**Why left-padding instead of right-padding?**  
The model reads the LSTM's *final hidden state* as the sequence representation. Left-padding ensures that the final hidden state reflects the actual last word of the input rather than a pad token.

**Why N-gram prefix expansion?**  
Expanding each sentence into all its prefixes significantly multiplies the effective training data without requiring additional text — particularly valuable for small corpora.

---

## Future Improvements

- Add top-k / nucleus (top-p) sampling for more diverse generation
- Stack multiple LSTM layers for deeper sequence modeling
- Add dropout regularization to reduce overfitting on small corpora
- Export model weights for inference without retraining
- Build a simple web UI with Gradio or Streamlit

---

## License

The Adventures of Sherlock Holmes is sourced from [Project Gutenberg](https://www.gutenberg.org/) and is in the public domain.
