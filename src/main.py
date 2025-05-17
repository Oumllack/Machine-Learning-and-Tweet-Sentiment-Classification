import logging
from pathlib import Path
from src.data.load_dataset import load_sentiment140_dataset, analyze_dataset, prepare_dataset
from src.data.preprocess import TweetPreprocessor
from src.models.train import TweetClassifier
from src.utils.helpers import setup_logging, create_directory, save_metrics

# Configuration du logger
logger = logging.getLogger(__name__)

def main():
    """Pipeline complet d'analyse et d'entraînement."""
    # Configuration
    setup_logging('logs/pipeline.log')
    create_directory('logs')
    create_directory('models')
    create_directory('results')
    
    # 1. Chargement et analyse des données
    logger.info("=== Étape 1: Chargement et analyse des données ===")
    dataset_path = 'data/training.1600000.processed.noemoticon.csv'
    df = load_sentiment140_dataset(dataset_path)
    analyze_dataset(df)
    
    # 2. Préparation des données
    logger.info("\n=== Étape 2: Préparation des données ===")
    train_df, test_df = prepare_dataset(df, sample_size=100000)
    
    # 3. Prétraitement des données
    logger.info("\n=== Étape 3: Prétraitement des données ===")
    preprocessor = TweetPreprocessor(language='english')  # Sentiment140 est en anglais
    
    logger.info("Prétraitement des données d'entraînement...")
    train_processed = preprocessor.preprocess_dataset(train_df, 'text', 'sentiment')
    
    logger.info("Prétraitement des données de test...")
    test_processed = preprocessor.preprocess_dataset(test_df, 'text', 'sentiment')
    
    # Sauvegarde des données prétraitées
    train_processed.to_csv('data/processed/train_processed.csv', index=False)
    test_processed.to_csv('data/processed/test_processed.csv', index=False)
    
    # 4. Entraînement et évaluation des modèles
    logger.info("\n=== Étape 4: Entraînement et évaluation des modèles ===")
    models = ['logistic_regression', 'naive_bayes', 'svm']
    results = {}
    
    for model_type in models:
        logger.info(f"\nEntraînement du modèle {model_type}...")
        
        # Initialisation et entraînement
        classifier = TweetClassifier(model_type=model_type)
        X_train = train_processed['processed_text']
        y_train = train_processed['sentiment']
        X_test = test_processed['processed_text']
        y_test = test_processed['sentiment']
        
        # Entraînement
        classifier.train(X_train, y_train)
        
        # Évaluation
        report = classifier.evaluate(X_test, y_test)
        results[model_type] = report
        
        # Sauvegarde du modèle
        classifier.save_model()
    
    # 5. Sauvegarde des résultats comparatifs
    logger.info("\n=== Étape 5: Sauvegarde des résultats comparatifs ===")
    save_metrics(results, 'results/model_comparison.json')
    
    # Affichage des résultats
    print("\n=== Résultats comparatifs des modèles ===")
    for model_type, report in results.items():
        print(f"\nModèle : {model_type}")
        print(f"Accuracy : {report['accuracy']:.3f}")
        print(f"F1-score moyen : {report['macro avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main() 