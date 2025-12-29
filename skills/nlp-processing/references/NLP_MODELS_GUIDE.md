# NLP Models Selection Guide

## Model Selection by Task

```
┌──────────────────────────────────────────────────────────────────────────┐
│ NLP Task                     │ Recommended Models                        │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Text Classification          │ BERT, RoBERTa, DistilBERT                 │
│ Sentiment Analysis           │ BERT, XLNet, DistilBERT                   │
│ Named Entity Recognition     │ BERT-NER, spaCy, Flair                    │
│ Question Answering           │ BERT-QA, RoBERTa, ALBERT                  │
│ Text Generation              │ GPT-2, GPT-3, T5, LLaMA                   │
│ Summarization                │ BART, T5, Pegasus                         │
│ Translation                  │ mBART, MarianMT, NLLB                     │
│ Semantic Similarity          │ Sentence-BERT, SimCSE                     │
│ Zero-shot Classification     │ BART-MNLI, DeBERTa-v3                     │
│ Token Classification         │ BERT, RoBERTa, LayoutLM                   │
└──────────────────────────────┴───────────────────────────────────────────┘
```

## Model Comparison

| Model | Parameters | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| DistilBERT | 66M | Fast | Good | Production, edge |
| BERT-base | 110M | Medium | Very Good | General NLP |
| BERT-large | 340M | Slow | Excellent | High accuracy |
| RoBERTa | 125M | Medium | Excellent | Most NLP tasks |
| ALBERT | 12M | Fast | Good | Mobile/edge |
| XLNet | 340M | Slow | Excellent | Long documents |
| GPT-2 | 1.5B | Slow | Excellent | Text generation |
| T5 | 220M-11B | Varies | Excellent | Multi-task |

## Preprocessing Decision Tree

```
Input Text
    │
    ▼
┌─────────────────────┐
│ Clean & Normalize   │ Remove HTML, URLs, special chars
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Tokenization        │ WordPiece/BPE/SentencePiece
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 Classical    Transformer
    │             │
    ▼             ▼
┌──────────┐  ┌──────────┐
│Stopwords │  │ Subword  │
│Stemming  │  │ Tokens   │
│TF-IDF    │  │ Attention│
└──────────┘  └──────────┘
```

## Embedding Methods Comparison

| Method | Type | Pros | Cons |
|--------|------|------|------|
| Bag of Words | Sparse | Simple | No semantics |
| TF-IDF | Sparse | Term importance | No context |
| Word2Vec | Dense | Semantic | Fixed vectors |
| GloVe | Dense | Good quality | Pre-trained only |
| FastText | Dense | Subwords | Larger files |
| BERT | Contextual | State-of-art | Compute heavy |
| Sentence-BERT | Sentence | Fast similarity | Task-specific |

## Fine-tuning Best Practices

### Learning Rate Schedule

```python
# Typical learning rates for fine-tuning
LEARNING_RATES = {
    'bert-base': 2e-5,
    'bert-large': 1e-5,
    'roberta': 2e-5,
    'distilbert': 5e-5,
    'gpt2': 5e-5
}

# Warmup steps (typically 6-10% of total)
warmup_steps = int(0.1 * total_steps)
```

### Data Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Classification | 1000 | 10,000+ |
| NER | 5000 | 50,000+ |
| QA | 10,000 | 100,000+ |
| Generation | 100,000 | 1M+ |

## Evaluation Metrics by Task

| Task | Primary Metrics |
|------|-----------------|
| Classification | Accuracy, F1, AUC-ROC |
| NER | Entity F1, Exact Match |
| QA | Exact Match, F1 |
| Generation | BLEU, ROUGE, Perplexity |
| Similarity | Spearman, Cosine Sim |
| Translation | BLEU, chrF, COMET |

## Common Architectures

### Encoder-only (BERT-style)
- Best for: Classification, NER, embeddings
- Examples: BERT, RoBERTa, DistilBERT

### Decoder-only (GPT-style)
- Best for: Text generation, completion
- Examples: GPT-2, GPT-3, LLaMA

### Encoder-Decoder (T5-style)
- Best for: Translation, summarization, multi-task
- Examples: T5, BART, mBART

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [Papers With Code NLP](https://paperswithcode.com/area/natural-language-processing)
- [spaCy Models](https://spacy.io/models)
- [NLTK Documentation](https://www.nltk.org/)
