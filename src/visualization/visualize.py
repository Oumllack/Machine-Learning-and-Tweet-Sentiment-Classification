import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_training_history(history: Dict[str, List[float]], title: str = 'Training History'):
    """
    Plot training and validation metrics history.
    
    Args:
        history (Dict[str, List[float]]): Training history dictionary
        title (str): Plot title
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = 'Confusion Matrix'):
    """
    Plot confusion matrix with custom labels.
    
    Args:
        cm (np.ndarray): Confusion matrix
        labels (List[str]): Class labels
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(feature_importance: Dict[str, float], title: str = 'Feature Importance'):
    """
    Plot feature importance scores.
    
    Args:
        feature_importance (Dict[str, float]): Dictionary of feature importance scores
        title (str): Plot title
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(20), x='Importance', y='Feature')
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_sentiment_distribution(sentiment_scores: List[float], title: str = 'Sentiment Distribution'):
    """
    Plot distribution of sentiment scores.
    
    Args:
        sentiment_scores (List[float]): List of sentiment scores
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiment_scores, bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='--', label='Neutral')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_model_comparison(results: Dict[str, Dict[str, float]], metric: str = 'accuracy'):
    """
    Plot comparison of different models based on a metric.
    
    Args:
        results (Dict[str, Dict[str, float]]): Dictionary of model results
        metric (str): Metric to compare (e.g., 'accuracy', 'f1-score')
    """
    # Prepare data
    models = list(results.keys())
    scores = [results[model][metric] for model in models]
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=scores)
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.ylabel(metric.capitalize())
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'model_comparison_{metric}.png')
    plt.close()

def plot_learning_curves(train_sizes: List[int], train_scores: List[float],
                        val_scores: List[float], title: str = 'Learning Curves'):
    """
    Plot learning curves showing model performance vs training size.
    
    Args:
        train_sizes (List[int]): List of training sizes
        train_scores (List[float]): List of training scores
        val_scores (List[float]): List of validation scores
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, val_scores, label='Validation Score')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_error_analysis(predictions: List[int], true_labels: List[int],
                       texts: List[str], n_samples: int = 5):
    """
    Plot examples of misclassified samples.
    
    Args:
        predictions (List[int]): Model predictions
        true_labels (List[int]): True labels
        texts (List[str]): Original texts
        n_samples (int): Number of examples to show
    """
    # Find misclassified samples
    misclassified = [(i, pred, true, text) for i, (pred, true, text) in
                    enumerate(zip(predictions, true_labels, texts))
                    if pred != true]
    
    if not misclassified:
        logger.info("No misclassified samples found!")
        return
    
    # Select random samples
    samples = np.random.choice(misclassified, min(n_samples, len(misclassified)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for ax, (idx, pred, true, text) in zip(axes, samples):
        ax.text(0.1, 0.5, f'Text: {text}\nPredicted: {pred}\nTrue: {true}',
                wrap=True, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Error Analysis - Misclassified Samples')
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'error_analysis.png')
    plt.close()

def main():
    """Test visualization functions with example data."""
    # Example training history
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4],
        'val_accuracy': [0.7, 0.75, 0.8]
    }
    plot_training_history(history)
    
    # Example confusion matrix
    cm = np.array([[100, 20], [30, 150]])
    plot_confusion_matrix(cm, ['Negative', 'Positive'])
    
    # Example feature importance
    feature_importance = {
        'word1': 0.3,
        'word2': 0.2,
        'word3': 0.1
    }
    plot_feature_importance(feature_importance)
    
    # Example sentiment distribution
    sentiment_scores = np.random.normal(0, 1, 1000)
    plot_sentiment_distribution(sentiment_scores)
    
    # Example model comparison
    results = {
        'Model1': {'accuracy': 0.8, 'f1-score': 0.75},
        'Model2': {'accuracy': 0.85, 'f1-score': 0.8}
    }
    plot_model_comparison(results)

if __name__ == "__main__":
    main() 