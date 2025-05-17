import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TweetClassifier:
    def __init__(self):
        """Initialise le classifieur de tweets avec Logistic Regression."""
        self.vectorizer = TfidfVectorizer(
            max_features=100000,  # Plus de features pour capturer plus de vocabulaire
            ngram_range=(1, 2),
            min_df=5,  # Ignorer les mots qui apparaissent moins de 5 fois
            max_df=0.95  # Ignorer les mots qui apparaissent dans plus de 95% des tweets
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,  # Utiliser tous les cœurs disponibles
            C=1.0,  # Régularisation L2
            solver='saga'  # Meilleur pour les grands datasets
        )
        
    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        logger.info("Vectorisation des textes...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        logger.info("Entraînement du modèle...")
        self.classifier.fit(X_train_vec, y_train)
        logger.info("Entraînement terminé")
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur les données de test."""
        logger.info("Évaluation du modèle...")
        
        # Vectorisation et prédictions
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vec)
        
        # Métriques
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Sauvegarde des résultats
        self._save_evaluation_results(report, cm)
        
        return report
    
    def _save_evaluation_results(self, report, cm):
        """Sauvegarde les résultats d'évaluation."""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Sauvegarde du rapport
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(results_dir / 'classification_report.csv')
        
        # Sauvegarde de la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie étiquette')
        plt.xlabel('Prédiction')
        plt.savefig(results_dir / 'confusion_matrix.png')
        plt.close()
    
    def save_model(self):
        """Sauvegarde le modèle entraîné."""
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        # Sauvegarde du vectorizer
        joblib.dump(self.vectorizer, model_dir / 'vectorizer.joblib')
        # Sauvegarde du classifieur
        joblib.dump(self.classifier, model_dir / 'classifier.joblib')
        logger.info("Modèle sauvegardé")
    
    def load_model(self):
        """Charge un modèle sauvegardé."""
        model_dir = Path('models')
        self.vectorizer = joblib.load(model_dir / 'vectorizer.joblib')
        self.classifier = joblib.load(model_dir / 'classifier.joblib')
        logger.info("Modèle chargé")

def main():
    """Fonction principale pour tester l'entraînement."""
    # Exemple d'utilisation
    sample_data = pd.DataFrame({
        'processed_text': [
            "film excellent acteur talentueux",
            "service client terrible attente longue",
            "produit qualité prix raisonnable",
            "application bug lente inutilisable"
        ],
        'sentiment': [1, 0, 1, 0]  # 1: positif, 0: négatif
    })
    
    # Initialisation et entraînement du modèle
    classifier = TweetClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        sample_data['processed_text'], sample_data['sentiment'], test_size=0.2, random_state=42
    )
    
    # Entraînement
    classifier.train(X_train, y_train)
    
    # Évaluation
    report = classifier.evaluate(X_test, y_test)
    print("\nRapport de classification :")
    print(pd.DataFrame(report).transpose())
    
    # Sauvegarde du modèle
    classifier.save_model()

if __name__ == "__main__":
    main() 