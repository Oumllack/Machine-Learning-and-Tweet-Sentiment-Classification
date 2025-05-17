import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (Union[str, Path]): Chemin vers le fichier de données
        **kwargs: Arguments additionnels pour pd.read_csv()
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Données chargées depuis {file_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {str(e)}")
        raise

def save_data(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Sauvegarde les données dans un fichier CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        file_path (Union[str, Path]): Chemin où sauvegarder les données
        **kwargs: Arguments additionnels pour pd.to_csv()
    """
    try:
        df.to_csv(file_path, **kwargs)
        logger.info(f"Données sauvegardées dans {file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données : {str(e)}")
        raise

def create_directory(path: Union[str, Path]) -> None:
    """
    Crée un répertoire s'il n'existe pas.
    
    Args:
        path (Union[str, Path]): Chemin du répertoire à créer
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Répertoire créé : {path}")

def save_metrics(metrics: Dict, file_path: Union[str, Path]) -> None:
    """
    Sauvegarde les métriques dans un fichier JSON.
    
    Args:
        metrics (Dict): Dictionnaire contenant les métriques
        file_path (Union[str, Path]): Chemin où sauvegarder les métriques
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logger.info(f"Métriques sauvegardées dans {file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des métriques : {str(e)}")
        raise

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Trace l'historique d'entraînement d'un modèle.
    
    Args:
        history (Dict[str, List[float]]): Historique d'entraînement
        save_path (Optional[Union[str, Path]]): Chemin où sauvegarder le graphique
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé dans {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, 
                         labels: List[str],
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Trace une matrice de confusion.
    
    Args:
        cm (np.ndarray): Matrice de confusion
        labels (List[str]): Labels des classes
        save_path (Optional[Union[str, Path]]): Chemin où sauvegarder le graphique
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Prédiction')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Matrice de confusion sauvegardée dans {save_path}")
    
    plt.show()

def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcule les poids des classes pour gérer le déséquilibre.
    
    Args:
        y (np.ndarray): Labels des classes
        
    Returns:
        Dict[int, float]: Dictionnaire des poids par classe
    """
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {
        i: total_samples / (len(class_counts) * count)
        for i, count in enumerate(class_counts)
    }
    return class_weights

def print_evaluation_metrics(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           labels: Optional[List[str]] = None) -> None:
    """
    Affiche les métriques d'évaluation.
    
    Args:
        y_true (np.ndarray): Labels réels
        y_pred (np.ndarray): Labels prédits
        labels (Optional[List[str]]): Labels des classes
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    print("\nMatrice de confusion :")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels or [str(i) for i in range(len(np.unique(y_true)))])

def setup_logging(log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Configure le logging avec un fichier optionnel.
    
    Args:
        log_file (Optional[Union[str, Path]]): Chemin du fichier de log
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info("Logging configuré") 