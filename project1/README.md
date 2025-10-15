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

## What the Notebook Contains

### 1. Data Loading and Preprocessing
- IMDB Movie Review Dataset loading
- Text cleaning and preprocessing pipeline
- Custom tokenizers for different model types
- Data splitting and DataLoader creation

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

## Project Objectives Fulfilled

- ✅ Text dataset from internet (IMDB)
- ✅ Sentiment analysis using time series-related neural networks
- ✅ RNN and LSTM comparison
- ✅ Transformer implementation
- ✅ Hyperparameter optimization
- ✅ Comprehensive report and analysis

## Contact

For questions or issues, please refer to the notebook documentation or create an issue.