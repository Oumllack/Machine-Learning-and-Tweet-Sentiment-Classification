import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_dataset import load_sentiment140_dataset
from src.preprocessing.tweet_preprocessor import TweetPreprocessor
from src.models.classical_models import TweetClassifier
from src.visualization.visualize import SentimentVisualizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Crée les répertoires nécessaires."""
    directories = ['data', 'models', 'results', 'logs', 'results/visualizations']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory created: {directory}")

def main():
    """Fonction principale pour le pipeline d'analyse de sentiment."""
    try:
        # Setup
        setup_directories()
        logger.info("Starting sentiment analysis pipeline...")
        
        # Charger les données
        logger.info("Loading dataset...")
        try:
            df = load_sentiment140_dataset('data/training.1600000.processed.noemoticon.csv')
            logger.info(f"Dataset loaded successfully: {len(df)} tweets")
            
            # Vérifier la distribution des classes
            class_dist = df['sentiment'].value_counts()
            logger.info(f"Class distribution in full dataset:\n{class_dist}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Prétraitement
        try:
            logger.info("Initializing preprocessor...")
            preprocessor = TweetPreprocessor(language='english')
            logger.info("Preprocessor initialized successfully")
            
            logger.info("Starting preprocessing...")
            df_processed = preprocessor.preprocess_dataset(df, 'text', 'sentiment')
            logger.info("Data preprocessing completed successfully")
            
            # Vérifier la distribution des classes
            class_dist = df_processed['sentiment'].value_counts()
            logger.info(f"Class distribution after preprocessing:\n{class_dist}")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
        
        # Entraînement et évaluation
        try:
            logger.info("Starting model training...")
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                df_processed['processed_text'],
                df_processed['sentiment'],
                test_size=0.2,
                random_state=42
            )
            
            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Test set size: {len(X_test)}")
            
            # Entraînement du modèle
            classifier = TweetClassifier()
            classifier.train(X_train, y_train)
            
            # Évaluation
            report = classifier.evaluate(X_test, y_test)
            logger.info("\nClassification Report:")
            logger.info(pd.DataFrame(report).transpose())
            
            # Sauvegarde du modèle
            classifier.save_model()
            logger.info("Model training completed successfully")
            
            # Génération des visualisations
            logger.info("Generating visualizations...")
            visualizer = SentimentVisualizer()
            
            # Obtenir les prédictions pour les visualisations
            X_test_vec = classifier.vectorizer.transform(X_test)
            y_pred = classifier.classifier.predict(X_test_vec)
            y_pred_proba = classifier.classifier.predict_proba(X_test_vec)[:, 1]
            
            # Générer toutes les visualisations
            visualizer.generate_all_visualizations(
                df_processed,
                y_test,
                y_pred,
                y_pred_proba,
                classifier.vectorizer,
                classifier.classifier
            )
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 