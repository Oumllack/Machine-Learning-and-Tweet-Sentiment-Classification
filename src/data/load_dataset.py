import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple
from src.utils.helpers import setup_logging, create_directory

# Configuration du logging
logger = logging.getLogger(__name__)

def load_sentiment140_dataset(file_path: str) -> pd.DataFrame:
    """
    Charge le dataset Sentiment140.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    # Les colonnes du dataset Sentiment140
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Chargement des données
    df = pd.read_csv(file_path, 
                     encoding='latin-1',
                     names=columns,
                     low_memory=False)
    
    # Conversion de la cible (0: négatif, 4: positif)
    df['sentiment'] = df['target'].map({0: 0, 4: 1})  # 0: négatif, 1: positif
    
    # Sélection des colonnes pertinentes
    df = df[['text', 'sentiment', 'date']]
    
    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def analyze_dataset(df: pd.DataFrame) -> None:
    """
    Analyse le dataset et affiche des statistiques.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    """
    print("\n=== Statistiques du Dataset ===")
    print(f"Nombre total de tweets : {len(df):,}")
    print(f"Période : de {df['date'].min()} à {df['date'].max()}")
    print("\nDistribution des sentiments :")
    print(df['sentiment'].value_counts(normalize=True).round(3))
    
    # Statistiques sur la longueur des tweets
    df['tweet_length'] = df['text'].str.len()
    print("\nStatistiques sur la longueur des tweets :")
    print(df['tweet_length'].describe().round(2))
    
    # Sauvegarde des statistiques
    stats = {
        'total_tweets': len(df),
        'date_min': df['date'].min().isoformat(),
        'date_max': df['date'].max().isoformat(),
        'sentiment_distribution': df['sentiment'].value_counts(normalize=True).to_dict(),
        'tweet_length_stats': df['tweet_length'].describe().to_dict()
    }
    
    # Création du dossier results s'il n'existe pas
    create_directory('results')
    
    # Sauvegarde des statistiques
    import json
    with open('results/dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

def prepare_dataset(df: pd.DataFrame, sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prépare le dataset pour l'entraînement.
    
    Args:
        df (pd.DataFrame): DataFrame complet
        sample_size (int, optional): Taille de l'échantillon à utiliser
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if sample_size:
        # Échantillonnage stratifié
        df = df.groupby('sentiment').apply(
            lambda x: x.sample(min(len(x), sample_size // 2))
        ).reset_index(drop=True)
    
    # Mélange des données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Séparation train/test (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    return train_df, test_df

def main():
    """Fonction principale."""
    # Chemin vers le dataset
    dataset_path = 'data/training.1600000.processed.noemoticon.csv'
    
    # Chargement du dataset
    print("Chargement du dataset...")
    df = load_sentiment140_dataset(dataset_path)
    
    # Analyse du dataset
    print("\nAnalyse du dataset...")
    analyze_dataset(df)
    
    # Préparation des données (utilisation d'un échantillon de 100 000 tweets)
    print("\nPréparation des données...")
    train_df, test_df = prepare_dataset(df, sample_size=100000)
    
    # Sauvegarde des datasets
    create_directory('data/processed')
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"\nDatasets sauvegardés :")
    print(f"- Train : {len(train_df):,} tweets")
    print(f"- Test  : {len(test_df):,} tweets")

if __name__ == "__main__":
    main() 