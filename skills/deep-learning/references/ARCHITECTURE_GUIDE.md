# Deep Learning Architecture Guide

## Neural Network Architectures Decision Tree

```
START: What type of data?
│
├─► Tabular Data
│   └─► MLP (Multi-Layer Perceptron)
│       • 2-5 hidden layers
│       • ReLU activation
│       • Dropout for regularization
│
├─► Image Data
│   ├─► Classification → CNN Architectures
│   │   ├─► Small dataset → Transfer Learning (ResNet, EfficientNet)
│   │   ├─► Medium dataset → Custom CNN or Fine-tuned pretrained
│   │   └─► Large dataset → Train from scratch or Vision Transformer
│   │
│   ├─► Object Detection
│   │   ├─► Real-time → YOLO, SSD
│   │   └─► High accuracy → Faster R-CNN, Mask R-CNN
│   │
│   └─► Segmentation
│       ├─► Semantic → U-Net, DeepLab
│       └─► Instance → Mask R-CNN
│
├─► Sequential Data (Time Series, Text)
│   ├─► Short sequences → 1D CNN
│   ├─► Long sequences → LSTM, GRU
│   └─► Attention needed → Transformer
│
├─► Text Data (NLP)
│   ├─► Classification → BERT, RoBERTa
│   ├─► Generation → GPT, T5
│   └─► Embeddings → Word2Vec, FastText, Sentence-BERT
│
└─► Graph Data
    └─► GNN (Graph Neural Networks)
        ├─► Node classification → GCN, GAT
        └─► Graph classification → Graph-level pooling
```

## Architecture Comparison Table

| Architecture | Best For | Params | Training Time | Accuracy |
|--------------|----------|--------|---------------|----------|
| ResNet-50 | General image tasks | 25M | Medium | High |
| EfficientNet-B0 | Mobile/Edge | 5.3M | Fast | High |
| VGG-16 | Feature extraction | 138M | Slow | Good |
| MobileNetV3 | Mobile deployment | 5.4M | Fast | Good |
| ViT-Base | Large datasets | 86M | Slow | Very High |
| BERT-Base | NLP tasks | 110M | Slow | Very High |
| GPT-2 | Text generation | 1.5B | Very Slow | Very High |

## Layer Selection Guide

### Activation Functions

| Function | Use Case | Pros | Cons |
|----------|----------|------|------|
| ReLU | Hidden layers (default) | Fast, no vanishing gradient | Dead neurons |
| LeakyReLU | Alternative to ReLU | No dead neurons | Slightly slower |
| GELU | Transformers | Smooth, better gradients | Computationally expensive |
| Sigmoid | Binary output | Bounded (0,1) | Vanishing gradient |
| Softmax | Multi-class output | Probability distribution | Numerical instability |
| Tanh | RNN hidden states | Bounded (-1,1) | Vanishing gradient |

### Regularization Techniques

```
Technique       │ When to Use                    │ Typical Values
────────────────┼────────────────────────────────┼─────────────────
Dropout         │ Fully connected layers         │ 0.2-0.5
L2 (Weight Dec) │ Always recommended             │ 1e-4 to 1e-2
Batch Norm      │ After conv/dense, before act   │ momentum=0.99
Layer Norm      │ Transformers, RNNs             │ eps=1e-6
Data Augment    │ Limited training data          │ Task-specific
Early Stopping  │ Prevent overfitting            │ patience=5-10
```

## Transfer Learning Strategy

```python
# Strategy Selection Based on Dataset Size

Dataset Size vs Domain Similarity Matrix:
                    │ Similar Domain │ Different Domain
────────────────────┼────────────────┼──────────────────
Small Dataset       │ Fine-tune      │ Fine-tune deeper
(< 1K samples)      │ top layers     │ layers + augment
────────────────────┼────────────────┼──────────────────
Medium Dataset      │ Fine-tune      │ Fine-tune all
(1K - 10K samples)  │ all layers     │ with low LR
────────────────────┼────────────────┼──────────────────
Large Dataset       │ Fine-tune or   │ Train from
(> 10K samples)     │ train scratch  │ scratch
```

## Hyperparameter Guidelines

### Learning Rate

```
Model Type          │ Initial LR      │ Schedule
────────────────────┼─────────────────┼──────────────────
CNN from scratch    │ 0.001 - 0.01    │ StepLR or Cosine
Fine-tuning         │ 0.0001 - 0.001  │ ReduceOnPlateau
Transformers        │ 1e-5 - 5e-5     │ Linear warmup
GANs                │ 0.0001 - 0.0002 │ Fixed or decay
```

### Batch Size

```
Memory Available    │ Recommended Batch Size
────────────────────┼─────────────────────────
8 GB GPU            │ 16 - 32 (images)
16 GB GPU           │ 32 - 64 (images)
24+ GB GPU          │ 64 - 128 (images)
Gradient Accum      │ Effective batch = batch × accum_steps
```

## Common Mistakes and Solutions

| Mistake | Symptom | Solution |
|---------|---------|----------|
| Too high LR | Loss explodes/oscillates | Reduce LR by 10x |
| Too low LR | Very slow convergence | Increase LR, use scheduler |
| No normalization | Unstable training | Add BatchNorm/LayerNorm |
| Too deep network | Vanishing gradients | Add skip connections |
| Small dataset | Overfitting | Augmentation, transfer learning |
| Wrong initialization | Dead/saturated neurons | Use He/Xavier init |

## Model Debugging Checklist

```markdown
□ Data pipeline verified (visualize samples)
□ Model can overfit single batch
□ Gradients flowing (not NaN or zero)
□ Learning rate appropriate (loss decreasing)
□ Validation metrics improving
□ No data leakage (train/val properly split)
□ Augmentation not too aggressive
□ Class imbalance addressed
```

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Papers With Code - State of the Art](https://paperswithcode.com/)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)
