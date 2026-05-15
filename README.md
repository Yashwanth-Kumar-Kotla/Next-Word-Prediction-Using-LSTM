# Next Word Prediction Using LSTM

A PyTorch LSTM language model that predicts the next word in a sentence, trained on two distinct corpora and served via a Streamlit web app.

**Live demo:** [nextt-word-predictor.streamlit.app](https://nextt-word-predictor.streamlit.app/)

---

## Architecture

```
Input Tokens
     │
     ▼
Embedding Layer   (vocab_size → 100 dims)
     │
     ▼
LSTM Layer        (100 → 150 hidden units, batch_first=True)
     │
     ▼  (final hidden state)
Linear Layer      (150 → vocab_size)
     │
     ▼
Argmax → Predicted Next Word
```

The model reads the LSTM's final hidden state as the sequence representation and predicts a single next word per forward pass. During inference, predictions are chained autoregressively to generate multi-word continuations.

---

## Training

Two models were trained — the deployed app uses the Sherlock Holmes model.

| Corpus | Vocabulary | Training Sequences | Epochs | Final Loss | Accuracy |
|---|---|---|---|---|---|
| Custom (40 sentences) | 222 | 302 | 50 | 0.163 | **96.36%** |
| Sherlock Holmes (full novel) | 9,398 | 123,765 | 40 | 0.711 | **86.00%** |

### How It Works

1. **Tokenization** — Text is lowercased and tokenized with NLTK's `word_tokenize`
2. **Vocabulary** — A word-to-index mapping is built from token frequencies via `Counter`; unknown tokens map to index `0` (`"unk"`)
3. **Sequence Construction** — Each sentence is expanded into all its prefixes (`[w1]→w2`, `[w1,w2]→w3`, …), multiplying effective training data
4. **Padding** — Sequences are left-padded with zeros to the length of the longest sequence, so the final LSTM hidden state always reflects the last real token
5. **Training** — Adam optimizer (`lr=0.001`) + CrossEntropyLoss; CUDA-accelerated when available
6. **Inference** — Input text is numericalized, left-padded, and fed to the model; argmax over output logits gives the next word

---

## Project Structure

```
├── app.py                                                      # Streamlit web app
├── model.py                                                    # LSTMNN class definition
├── model.pth                                                   # Trained weights (Sherlock Holmes corpus)
├── vocab.pkl                                                   # Word-to-index vocabulary
├── max_len.pkl                                                 # Max sequence length for padding
├── requirements.txt                                            # Python dependencies
├── document.txt                                                # The Adventures of Sherlock Holmes (Project Gutenberg)
├── Next_Word_Prediction_using_LSTM.ipynb                       # Training notebook — custom corpus
└── Next_Word_Prediction_using_LSTM on kaggle dataset.ipynb    # Training notebook — Sherlock Holmes corpus
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

```python
# NLTK data (downloaded automatically by app.py, or run manually)
import nltk
nltk.download('punkt_tab')
```

---

## Key Design Decisions

**Why LSTM over a simple RNN?**
LSTMs use gating mechanisms (input, forget, output gates) to retain long-range dependencies — critical when the relevant context is many words back.

**Why left-padding instead of right-padding?**
The model uses the final hidden state as its sequence representation. Left-padding ensures the final hidden state reflects the last real word, not a pad token.

**Why N-gram prefix expansion?**
Expanding each sentence into all its prefixes significantly multiplies effective training data without requiring additional text — especially valuable for small corpora.

---

## License

*The Adventures of Sherlock Holmes* is sourced from [Project Gutenberg](https://www.gutenberg.org/) and is in the public domain.
