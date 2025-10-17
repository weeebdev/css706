# Sentiment Analysis Project: RNN vs LSTM vs Transformer

A comprehensive comparison of different neural network architectures for sentiment analysis on the IMDB movie review dataset using Jupyter Notebook.

## Project Overview

This project implements and compares four different neural network architectures for binary sentiment classification:

1. **Recurrent Neural Network (RNN)** - Basic sequential processing
2. **Long Short-Term Memory (LSTM)** - Enhanced RNN with attention mechanism
3. **Pre-trained Transformer (BERT)** - State-of-the-art pre-trained model
4. **Custom Transformer** - Custom implementation without pre-training

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Project
Open `Sentiment_Analysis_Project.ipynb` in Jupyter Lab/Notebook and run all cells sequentially.

## Project Structure

```
project1/
├── Sentiment_Analysis_Project.ipynb    # Main comprehensive notebook
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Features

- ✅ **Complete Implementation** - All four architectures in one notebook
- ✅ **Interactive Execution** - Run cells step by step
- ✅ **Hyperparameter Optimization** - Automated parameter tuning
- ✅ **Comprehensive Analysis** - Performance comparison and visualizations
- ✅ **Educational Value** - Well-documented and explained code
- ✅ **Overfitting Prevention** - Advanced regularization and data augmentation techniques
- ✅ **Memory Efficiency** - Sequential training with model unloading for 10GB RAM constraint

## What the Notebook Contains

### 1. Data Loading and Preprocessing
- IMDB Movie Review Dataset loading
- Text cleaning and preprocessing pipeline
- Custom tokenizers for different model types
- Data splitting and DataLoader creation
- **Data Augmentation** - Text noise injection and word dropout to prevent overfitting

### 2. Model Architectures
- **RNN Model**: Basic recurrent neural network with bidirectional processing
- **LSTM Model**: Long Short-Term Memory with attention mechanism
- **BERT Model**: Pre-trained transformer for transfer learning
- **Custom Transformer**: Custom implementation without pre-training

### 3. Training and Evaluation
- Comprehensive training framework
- Early stopping and learning rate scheduling
- Training history visualization
- Confusion matrix plotting
- **Advanced Regularization** - High dropout, weight decay, gradient clipping
- **Overfitting Prevention** - Smaller model architectures and conservative training

### 4. Hyperparameter Optimization
- Simple optimization framework for LSTM
- Parameter search space definition
- Optimization progress tracking

### 5. Analysis and Conclusions
- Performance comparison across all models
- Model complexity analysis
- Efficiency metrics (accuracy per parameter)
- Radar charts and visualizations
- Detailed conclusions and recommendations

## Usage

1. **Open the notebook** in Jupyter Lab/Notebook
2. **Run cells sequentially** to execute the entire project
3. **Modify parameters** as needed for experimentation
4. **View results** including training curves, confusion matrices, and performance comparisons

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Scikit-learn
- Matplotlib/Seaborn
- Pandas/NumPy
- NLTK
- Datasets

## Results

The notebook will automatically:
- Load and preprocess the IMDB dataset
- Train all four models
- Compare their performance
- Generate comprehensive analysis and visualizations
- Provide detailed conclusions and recommendations

## Data Quality and Overfitting Prevention

This project implements comprehensive fixes for data overlap and overfitting issues:

### 1. **Data Overlap Fix** ✅
**Problem Identified:**
- Text preprocessing converted different original texts to identical cleaned texts
- This caused 3 overlapping samples between train/validation/test splits
- Sklearn's train_test_split was working correctly, but preprocessing introduced duplicates

**Solution Implemented:**
- Remove overlapping samples from validation and test sets after preprocessing
- Maintain training set integrity to preserve maximum training data
- Verify zero overlaps using set intersections
- Document expected sample count changes (e.g., 500 → 497 samples in validation)

### 2. **Overfitting Prevention Techniques** 🛡️

#### Model Architecture:
- **Small model size**: 16-32 dimensional embeddings, 16-32 hidden dimensions
- **Single layer**: Reduce model capacity to match small dataset (2000 samples)
- **Unidirectional**: Use unidirectional RNN/LSTM instead of bidirectional
- **Parameter count**: ~80K parameters instead of millions

#### Strong Regularization:
- **High dropout (0.6)**: Applied at multiple layers (embedding, recurrent, classifier)
- **Label smoothing (0.1)**: Prevents overconfident predictions
- **Weight decay (1e-2)**: Strong L2 regularization on model parameters
- **Gradient clipping (1.0)**: Prevent exploding gradients

#### Training Strategy:
- **Low learning rate (0.0005)**: Slower, more stable training
- **Aggressive early stopping (patience=3)**: Stop if validation loss doesn't improve
- **LR scheduling**: Reduce learning rate when validation loss plateaus
- **Overfitting detection**: Monitor train-val gap and stop if gap > 20%

#### Expected Healthy Results:
- ✅ Train accuracy: 70-85% (NOT 100%)
- ✅ Validation accuracy: 65-80% (NOT 100%)
- ✅ Train-val gap: < 10% (healthy generalization)
- ✅ No data leakage between splits

### 3. **Why Previous Results Were Problematic** ⚠️
- Models achieved 100% train AND validation accuracy → Clear overfitting
- Models memorized training data instead of learning patterns
- No generalization to truly unseen data
- Data overlap artificially inflated validation performance

### 4. Memory Efficiency (10GB RAM Constraint)
- **Tiny Model Architectures**: 16-32 dimensional embeddings
- **Sequential Training**: Train one model at a time, unload before next
- **Memory Monitoring**: Real-time RAM usage tracking and cleanup
- **Reduced Dataset**: 2,000 training + 500 val + 500 test samples
- **Small Batch Size**: 16 samples per batch
- **Truncated Sequences**: Maximum 64 tokens per text

## Project Objectives Fulfilled

- ✅ Text dataset from internet (IMDB)
- ✅ Sentiment analysis using time series-related neural networks
- ✅ RNN and LSTM comparison
- ✅ Transformer implementation
- ✅ Hyperparameter optimization
- ✅ Comprehensive report and analysis
- ✅ **Overfitting prevention and regularization techniques**

## Contact

For questions or issues, please refer to the notebook documentation or create an issue.