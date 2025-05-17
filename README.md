# Twitter Sentiment Analysis

A robust machine learning pipeline for sentiment analysis of tweets, achieving state-of-the-art performance on the Sentiment140 dataset.

## 📊 Project Overview

This project implements a complete machine learning pipeline for sentiment analysis of tweets, using the Sentiment140 dataset (1.6M tweets). The system achieves high accuracy in classifying tweets as positive or negative, with a balanced approach to both classes.

## 🎯 Key Features

- **Large-Scale Processing**: Handles 1.6M tweets efficiently
- **High Performance**: 82.2% accuracy on test set
- **Balanced Classification**: Equal precision for both positive and negative sentiments
- **Production-Ready**: Complete pipeline from data loading to model deployment
- **Memory Efficient**: Batch processing for large datasets
- **Comprehensive Logging**: Detailed logging of all pipeline steps

## 🛠️ Technologies & Libraries

### Core Technologies
- Python 3.9+
- scikit-learn
- pandas
- numpy
- joblib

### Key Libraries
- **Text Processing**: 
  - Regular Expressions (re)
  - NLTK (for future enhancements)
- **Machine Learning**:
  - scikit-learn (TfidfVectorizer, LogisticRegression)
  - Custom preprocessing pipeline
- **Data Handling**:
  - pandas for efficient data manipulation
  - numpy for numerical operations
- **Visualization**:
  - seaborn
  - matplotlib
- **Utilities**:
  - tqdm for progress tracking
  - logging for comprehensive pipeline monitoring

## 📈 Machine Learning Pipeline

### 1. Data Processing
- **Dataset**: Sentiment140 (1.6M tweets)
  - 800K positive tweets
  - 800K negative tweets
- **Preprocessing Steps**:
  - URL removal
  - @mentions removal
  - Hashtag symbol removal (keeping text)
  - Special character removal
  - Case normalization
  - Whitespace normalization
- **Processing Statistics**:
  - Total tweets processed: 1,600,000
  - Empty tweets removed: 3,170 (0.2%)
  - Final dataset size: 1,596,830 tweets
  - Processing speed: ~70-75K tweets/second
  - Batch size: 100K tweets

### 2. Feature Engineering
- **Vectorization**: TF-IDF with optimized parameters
  - Max features: 100,000
  - N-gram range: (1, 2)
  - Min document frequency: 5
  - Max document frequency: 95%
- **Vocabulary Size**: 100,000 unique tokens
- **Feature Types**: Unigrams and bigrams

### 3. Model Architecture
- **Algorithm**: Logistic Regression
- **Optimization**:
  - Solver: 'saga' (optimized for large datasets)
  - Regularization: L2 (C=1.0)
  - Max iterations: 1000
  - Parallel processing: All CPU cores
- **Training Parameters**:
  - Train/Test split: 80/20
  - Random state: 42
  - Training set size: 1,277,464 tweets
  - Test set size: 319,366 tweets

### 4. Performance Metrics
- **Overall Accuracy**: 82.2%
- **Class-wise Performance**:
  - Negative Class (0):
    * Precision: 82.7%
    * Recall: 81.3%
    * F1-score: 82.0%
  - Positive Class (1):
    * Precision: 81.7%
    * Recall: 83.1%
    * F1-score: 82.4%
- **Macro Average**:
  * Precision: 82.2%
  * Recall: 82.2%
  * F1-score: 82.2%

### 5. Training Process
- **Vectorization Time**: ~52 seconds
- **Training Time**: ~29 seconds
- **Total Pipeline Time**: ~2 minutes
- **Memory Usage**: Optimized through batch processing

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.9
pip install -r requirements.txt
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
Run the complete pipeline:
```bash
python -m src.main
```

## 📁 Project Structure
```
twitter-sentiment-analysis/
├── data/
│   └── training.1600000.processed.noemoticon.csv
├── models/
│   ├── vectorizer.joblib
│   └── classifier.joblib
├── results/
│   ├── classification_report.csv
│   └── confusion_matrix.png
├── src/
│   ├── data/
│   │   └── load_dataset.py
│   ├── preprocessing/
│   │   └── tweet_preprocessor.py
│   ├── models/
│   │   └── classical_models.py
│   └── main.py
├── logs/
│   └── pipeline.log
├── requirements.txt
└── README.md
```

## 📊 Results Visualization
The pipeline generates:
- Classification report (CSV)
- Confusion matrix visualization
- Detailed logging of all steps

## 🔄 Future Improvements
- Implement deep learning models (BERT, RoBERTa)
- Add cross-validation
- Implement hyperparameter tuning
- Add support for multi-class sentiment
- Implement real-time prediction API

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 