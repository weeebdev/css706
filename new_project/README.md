# Sentiment Analysis using Time Series Neural Networks

## Project Overview

This project implements and compares three different neural network architectures for sentiment analysis on text data:
1. **Simple RNN** (Recurrent Neural Network)
2. **LSTM** (Long Short-Term Memory) including Bidirectional LSTM
3. **Transformer** (Self-Attention based architecture)

## Dataset

- **Source**: IMDB Movie Reviews
- **Size**: 50,000 reviews (25,000 train + 25,000 test)
- **Labels**: Binary classification (Positive/Negative)
- **Vocabulary**: Top 10,000 most frequent words
- **Sequence Length**: Maximum 200 tokens

## Hardware Requirements

- **RAM**: 10GB minimum
- **GPU**: Optional but recommended for faster training
- **Storage**: ~2GB for models and datasets

## Installation

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Sentiment_Analysis_Neural_Networks.ipynb
```

2. Run all cells sequentially to:
   - Load and explore the IMDB dataset
   - Preprocess the data
   - Build and train RNN, LSTM, BiLSTM, and Transformer models
   - Perform hyperparameter optimization
   - Compare model performance
   - Analyze overfitting/underfitting
   - Generate comprehensive report

## Project Structure

```
new_project/
├── Sentiment_Analysis_Neural_Networks.ipynb  # Main notebook
├── README.md                                  # This file
├── rnn_sentiment_model.h5                    # Saved RNN model
├── lstm_sentiment_model.h5                   # Saved LSTM model
├── bilstm_sentiment_model.h5                 # Saved BiLSTM model
├── transformer_sentiment_model.h5            # Saved Transformer model
├── model_comparison_results.csv              # Performance metrics
├── hyperparameter_search_results.csv         # Hyperparameter tuning results
├── sentiment_analysis_report.txt             # Detailed report
├── model_config.json                         # Model configuration
└── *.png                                      # Visualization plots
```

## Key Features

### 1. Model Architectures

#### Simple RNN
- Basic recurrent architecture
- Fastest training time
- Suitable for quick prototyping
- Limited ability to capture long-term dependencies

#### LSTM (Long Short-Term Memory)
- Solves vanishing gradient problem
- Better at capturing sequential patterns
- Good balance of speed and accuracy
- Available in both unidirectional and bidirectional variants

#### Transformer
- State-of-the-art architecture
- Parallel processing capabilities
- Self-attention mechanism
- Best for larger datasets

### 2. Overfitting Prevention Techniques

- **Dropout Layers**: Spatial dropout (0.3) and regular dropout
- **Early Stopping**: Monitors validation loss with patience=5
- **Learning Rate Scheduling**: ReduceLROnPlateau callback
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Data Splitting**: 80% train / 20% validation split

### 3. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC Score
- Training Time
- Parameter Count

## Results Summary

The notebook will generate:
- Training history plots (accuracy, loss, AUC) for each model
- Confusion matrices and ROC curves
- Comprehensive comparison visualizations
- Hyperparameter optimization results
- Overfitting/underfitting analysis
- Detailed text report

## Hyperparameter Optimization

The project includes grid search for BiLSTM with variations in:
- LSTM units: 32, 64, 128
- Dropout rates: 0.2, 0.3, 0.4
- Learning rates: 0.0001, 0.0005, 0.001

## Expected Performance

Typical results (may vary):
- **Simple RNN**: ~82-85% accuracy
- **LSTM**: ~86-88% accuracy
- **BiLSTM**: ~88-90% accuracy (Best)
- **Transformer**: ~87-89% accuracy

## Recommendations

### For Production:
- Use BiLSTM for best accuracy
- Implement model ensembling
- Monitor for data drift

### For Resource-Constrained Environments:
- Use simple RNN or LSTM
- Consider model quantization
- Reduce vocabulary size and sequence length

### For Further Improvement:
- Use pre-trained embeddings (GloVe, Word2Vec, BERT)
- Increase dataset size
- Try ensemble methods
- Fine-tune on domain-specific data

## Report Generation

The notebook automatically generates:
1. **sentiment_analysis_report.txt**: Comprehensive text report
2. **model_comparison_results.csv**: Tabular performance comparison
3. **hyperparameter_search_results.csv**: Optimization results
4. **Multiple PNG files**: Visualization plots

## Troubleshooting

### Out of Memory Errors:
- Reduce `BATCH_SIZE` (default: 64)
- Reduce `VOCAB_SIZE` (default: 10000)
- Reduce `MAX_LENGTH` (default: 200)
- Reduce model complexity (fewer units/layers)

### Slow Training:
- Enable GPU acceleration
- Reduce number of epochs
- Increase batch size (if memory allows)
- Use smaller models (RNN instead of LSTM/Transformer)

### Overfitting:
- Increase dropout rate
- Add more regularization
- Collect more training data
- Reduce model complexity

### Underfitting:
- Increase model complexity
- Train for more epochs
- Reduce dropout rate
- Improve feature engineering

## Key Insights

1. **BiLSTM** generally achieves the best accuracy due to bidirectional context understanding
2. **Transformers** require more data and computational resources but offer parallel processing
3. **Simple RNN** is fastest but struggles with long sequences
4. **Dropout and early stopping** are crucial for preventing overfitting
5. **Hyperparameter tuning** can significantly improve model performance

## Citation

If you use this code, please cite:
```
Sentiment Analysis using Time-Series Neural Networks
Dataset: IMDB Movie Reviews from Keras datasets
```

## License

This project is for educational purposes.

## Author

PhD Project - CSS706
Date: 2025

## Contact

For questions or issues, please refer to the comprehensive report generated by the notebook.

