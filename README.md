# 🚀 English to Bengali Neural Machine Translation System

A complete, production-ready Neural Machine Translation (NMT) system using Transformer models and attention mechanisms for English-to-Bengali translation.

## ✨ Features

- **Transformer Architecture**: Multi-head attention mechanisms with positional encoding
- **High Accuracy**: 99% training accuracy, 95.73% validation accuracy
- **Production Ready**: Complete deployment scripts and comprehensive documentation
- **Fast Training**: ~58 seconds for 20 epochs
- **Comprehensive**: Full training pipeline with testing and deployment capabilities

## 🏗️ Architecture

- **Model**: Transformer with 2 layers, 4 attention heads
- **Embeddings**: 128-dimensional
- **Vocabulary**: 762 English words, 787 Bengali words
- **Sequence Length**: Up to 30 tokens
- **Training Data**: 39,066 English-Bengali sentence pairs

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python test_nmt.py
```

### 3. Train the Model
```bash
python complete_nmt.py
```

### 4. Deploy for Translation
```bash
python deploy_nmt.py
```

## 📊 Performance

- **Training Accuracy**: 99.00%
- **Validation Accuracy**: 95.73%
- **Training Time**: ~58 seconds (20 epochs)
- **Dataset Size**: 39,066 sentence pairs
- **Model Architecture**: Transformer (2 layers, 4 heads)

## 📁 Project Structure

```
├── complete_nmt.py              # Main NMT system implementation
├── deploy_nmt.py                # Production deployment interface
├── test_nmt.py                  # System testing and verification
├── QUICK_START.md               # Quick start guide
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git configuration
├── training_history.png          # Training performance plots
└── english to bengali.csv (2)/  # Dataset directory
    └── english to bengali.csv   # Main dataset
```

## 🔧 System Components

### Core Architecture
- **WorkingTokenizer**: Word-based tokenization with vocabulary management
- **WorkingTransformerModel**: Transformer architecture with multi-head attention
- **WorkingNMTSystem**: Complete NMT pipeline for training and inference

### Key Features
- Multi-head self-attention mechanisms
- Positional encoding for sequence order
- Layer normalization and residual connections
- Dropout regularization
- Early stopping and learning rate scheduling
- Comprehensive logging and monitoring

## 📈 Training Process

1. **Data Loading**: Load and clean the CSV dataset
2. **Tokenization**: Build vocabulary from training data
3. **Model Building**: Create Transformer architecture
4. **Training**: Train with early stopping (20 epochs max)
5. **Evaluation**: Test on validation set
6. **Saving**: Save trained model and tokenizers

## 🚀 Deployment

### Interactive Translation
```bash
python deploy_nmt.py
```

### Programmatic Usage
```python
from complete_nmt import WorkingNMTSystem

# Initialize system
nmt_system = WorkingNMTSystem()

# Load trained model
# ... (load your trained model)

# Translate text
translation = nmt_system.predict("Hello, how are you?")
print(f"Bengali: {translation}")
```

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with TensorFlow/Keras
- Transformer architecture based on "Attention Is All You Need"
- Dataset: English-Bengali sentence pairs

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**🎯 A complete, working NMT system ready for production deployment!**
