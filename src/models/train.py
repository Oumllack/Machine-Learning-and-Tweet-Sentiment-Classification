import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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
    def __init__(self, model_type='logistic_regression'):
        """
        Initialise le classifieur de tweets.
        
        Args:
            model_type (str): Type de modèle à utiliser ('logistic_regression', 'naive_bayes', ou 'svm')
        """
        self.model_type = model_type
        self.pipeline = None
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Sélection du modèle
        if model_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif model_type == 'svm':
            self.classifier = LinearSVC(random_state=42)
        else:
            raise ValueError("Type de modèle non supporté")
        
        # Création du pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
    def prepare_data(self, df, text_column, target_column, test_size=0.2):
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df (pandas.DataFrame): DataFrame contenant les données
            text_column (str): Nom de la colonne contenant le texte
            target_column (str): Nom de la colonne cible
            test_size (float): Proportion des données de test
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X = df[text_column]
        y = df[target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def train(self, X_train, y_train):
        """
        Entraîne le modèle.
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
        """
        logger.info(f"Entraînement du modèle {self.model_type}...")
        self.pipeline.fit(X_train, y_train)
        logger.info("Entraînement terminé")
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        
        Args:
            X_test: Données de test
            y_test: Labels de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        logger.info("Évaluation du modèle...")
        
        # Prédictions
        y_pred = self.pipeline.predict(X_test)
        
        # Métriques
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Sauvegarde des résultats
        self._save_evaluation_results(report, cm, y_test, y_pred)
        
        return report
    
    def _save_evaluation_results(self, report, cm, y_test, y_pred):
        """
        Sauvegarde les résultats d'évaluation.
        
        Args:
            report (dict): Rapport de classification
            cm (numpy.ndarray): Matrice de confusion
            y_test: Labels réels
            y_pred: Labels prédits
        """
        # Création du dossier results s'il n'existe pas
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Sauvegarde du rapport de classification
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(results_dir / f'{self.model_type}_classification_report.csv')
        
        # Sauvegarde de la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de Confusion - {self.model_type}')
        plt.ylabel('Vraie étiquette')
        plt.xlabel('Prédiction')
        plt.savefig(results_dir / f'{self.model_type}_confusion_matrix.png')
        plt.close()
    
    def save_model(self, path='models'):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            path (str): Chemin où sauvegarder le modèle
        """
        model_dir = Path(path)
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f'{self.model_type}_model.joblib'
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Modèle sauvegardé à {model_path}")
    
    def load_model(self, path):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path (str): Chemin vers le modèle sauvegardé
        """
        self.pipeline = joblib.load(path)
        logger.info(f"Modèle chargé depuis {path}")

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
    classifier = TweetClassifier(model_type='logistic_regression')
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        sample_data, 'processed_text', 'sentiment'
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