"""
Analyse exploratoire des données de tweets pour la classification de sentiment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import spacy
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import logging
from pathlib import Path
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TweetAnalyzer:
    def __init__(self, data_path='data/training.1600000.processed.noemoticon.csv'):
        """Initialise l'analyseur de tweets."""
        self.data_path = data_path
        self.df = None
        self.nlp = spacy.load('en_core_web_md')
        self.stop_words = set(stopwords.words('english'))
        
        # Création des répertoires pour les visualisations
        self.viz_dir = Path('results/visualizations')
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Charge les données des tweets."""
        logger.info("Chargement des données...")
        self.df = pd.read_csv(self.data_path, 
                            encoding='latin-1',
                            names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Conversion des valeurs de sentiment (0, 4) en (0, 1)
        self.df['sentiment'] = self.df['sentiment'].map({0: 0, 4: 1})
        logger.info(f"Données chargées : {len(self.df)} tweets")
        return self.df
    
    def analyze_temporal_patterns(self):
        """Analyse les patterns temporels des tweets."""
        logger.info("Analyse des patterns temporels...")
        
        # Distribution des tweets par jour
        plt.figure(figsize=(15, 6))
        self.df.groupby(self.df['date'].dt.date).size().plot(kind='line')
        plt.title('Distribution des tweets par jour')
        plt.xlabel('Date')
        plt.ylabel('Nombre de tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tweets_over_time.png')
        plt.close()
        
        # Distribution des sentiments par jour
        plt.figure(figsize=(15, 6))
        daily_sentiment = self.df.groupby([self.df['date'].dt.date, 'sentiment']).size().unstack()
        daily_sentiment.plot(kind='line')
        plt.title('Distribution des sentiments par jour')
        plt.xlabel('Date')
        plt.ylabel('Nombre de tweets')
        plt.legend(['Négatif', 'Positif'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sentiment_over_time.png')
        plt.close()
        
    def analyze_text_length(self):
        """Analyse la distribution de la longueur des tweets."""
        logger.info("Analyse de la longueur des tweets...")
        
        self.df['tweet_length'] = self.df['text'].str.len()
        
        # Distribution de la longueur des tweets
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='tweet_length', hue='sentiment', 
                    multiple="stack", bins=50)
        plt.title('Distribution de la longueur des tweets par sentiment')
        plt.xlabel('Longueur du tweet')
        plt.ylabel('Nombre de tweets')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tweet_length_distribution.png')
        plt.close()
        
        # Statistiques de longueur par sentiment
        length_stats = self.df.groupby('sentiment')['tweet_length'].agg(['mean', 'std', 'min', 'max'])
        logger.info("\nStatistiques de longueur par sentiment :\n" + str(length_stats))
        
    def generate_wordclouds(self):
        """Génère des nuages de mots pour les tweets positifs et négatifs."""
        logger.info("Génération des nuages de mots...")
        
        # Prétraitement du texte
        def preprocess_text(text):
            # Conversion en minuscules et suppression des caractères spéciaux
            text = text.lower()
            text = ' '.join([word for word in text.split() if word.isalnum()])
            return text
        
        # Nuage de mots pour les tweets positifs
        positive_text = ' '.join(self.df[self.df['sentiment'] == 1]['text'].apply(preprocess_text))
        if positive_text.strip():  # Vérifie si le texte n'est pas vide
            wordcloud_positive = WordCloud(width=800, height=400, 
                                         background_color='white',
                                         max_words=100).generate(positive_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de mots - Tweets positifs')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'wordcloud_positive.png')
            plt.close()
        
        # Nuage de mots pour les tweets négatifs
        negative_text = ' '.join(self.df[self.df['sentiment'] == 0]['text'].apply(preprocess_text))
        if negative_text.strip():  # Vérifie si le texte n'est pas vide
            wordcloud_negative = WordCloud(width=800, height=400,
                                         background_color='white',
                                         max_words=100).generate(negative_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de mots - Tweets négatifs')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'wordcloud_negative.png')
            plt.close()
        
    def analyze_common_words(self):
        """Analyse les mots les plus communs par sentiment."""
        logger.info("Analyse des mots les plus communs...")
        
        def get_common_words(texts, n=20):
            # Prétraitement du texte
            words = ' '.join(texts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2 and w.isalnum()]
            return Counter(words).most_common(n)
        
        # Mots communs pour les tweets positifs
        positive_words = get_common_words(self.df[self.df['sentiment'] == 1]['text'])
        if positive_words:  # Vérifie si la liste n'est pas vide
            plt.figure(figsize=(12, 6))
            words, counts = zip(*positive_words)
            plt.barh(words, counts)
            plt.title('20 mots les plus communs dans les tweets positifs')
            plt.xlabel('Fréquence')
            plt.ylabel('Mots')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'common_words_positive.png')
            plt.close()
        
        # Mots communs pour les tweets négatifs
        negative_words = get_common_words(self.df[self.df['sentiment'] == 0]['text'])
        if negative_words:  # Vérifie si la liste n'est pas vide
            plt.figure(figsize=(12, 6))
            words, counts = zip(*negative_words)
            plt.barh(words, counts)
            plt.title('20 mots les plus communs dans les tweets négatifs')
            plt.xlabel('Fréquence')
            plt.ylabel('Mots')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'common_words_negative.png')
            plt.close()
        
    def analyze_user_activity(self):
        """Analyse l'activité des utilisateurs."""
        logger.info("Analyse de l'activité des utilisateurs...")
        
        # Nombre de tweets par utilisateur
        user_activity = self.df['user'].value_counts()
        
        plt.figure(figsize=(12, 6))
        user_activity.head(20).plot(kind='bar')
        plt.title('Top 20 utilisateurs les plus actifs')
        plt.xlabel('Utilisateur')
        plt.ylabel('Nombre de tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'user_activity.png')
        plt.close()
        
        # Distribution du nombre de tweets par utilisateur
        plt.figure(figsize=(12, 6))
        sns.histplot(user_activity, bins=50)
        plt.title('Distribution du nombre de tweets par utilisateur')
        plt.xlabel('Nombre de tweets')
        plt.ylabel('Nombre d\'utilisateurs')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'user_activity_distribution.png')
        plt.close()
        
    def analyze_sentiment_patterns(self):
        """Analyse les patterns de sentiment."""
        logger.info("Analyse des patterns de sentiment...")
        
        # Distribution des sentiments
        plt.figure(figsize=(8, 6))
        self.df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribution des sentiments')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sentiment_distribution.png')
        plt.close()
        
        # Analyse des sentiments par utilisateur
        user_sentiment = self.df.groupby('user')['sentiment'].agg(['mean', 'count'])
        user_sentiment = user_sentiment[user_sentiment['count'] >= 10]  # Utilisateurs avec au moins 10 tweets
        
        plt.figure(figsize=(12, 6))
        sns.histplot(user_sentiment['mean'], bins=50)
        plt.title('Distribution du sentiment moyen par utilisateur')
        plt.xlabel('Sentiment moyen (0=Négatif, 1=Positif)')
        plt.ylabel('Nombre d\'utilisateurs')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'user_sentiment_distribution.png')
        plt.close()
        
    def generate_summary_report(self):
        """Génère un rapport sommaire des analyses."""
        logger.info("Génération du rapport sommaire...")
        
        summary = {
            'nombre_total_tweets': int(len(self.df)),
            'periode_analyse': {
                'debut': self.df['date'].min().strftime('%Y-%m-%d'),
                'fin': self.df['date'].max().strftime('%Y-%m-%d')
            },
            'distribution_sentiment': {
                'negatif': int(len(self.df[self.df['sentiment'] == 0])),
                'positif': int(len(self.df[self.df['sentiment'] == 1]))
            },
            'statistiques_longueur': {
                'moyenne': float(self.df['tweet_length'].mean()),
                'mediane': float(self.df['tweet_length'].median()),
                'min': int(self.df['tweet_length'].min()),
                'max': int(self.df['tweet_length'].max())
            },
            'statistiques_utilisateurs': {
                'nombre_total': int(self.df['user'].nunique()),
                'moyenne_tweets_par_utilisateur': float(len(self.df) / self.df['user'].nunique())
            }
        }
        
        # Sauvegarde du rapport
        with open(self.viz_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
            
        logger.info("Rapport sommaire généré et sauvegardé")
        
    def run_full_analysis(self):
        """Exécute l'ensemble des analyses."""
        logger.info("Démarrage de l'analyse complète...")
        
        self.load_data()
        self.analyze_temporal_patterns()
        self.analyze_text_length()
        self.generate_wordclouds()
        self.analyze_common_words()
        self.analyze_user_activity()
        self.analyze_sentiment_patterns()
        self.generate_summary_report()
        
        logger.info("Analyse complète terminée. Les visualisations sont disponibles dans le dossier 'results/visualizations'")

if __name__ == '__main__':
    analyzer = TweetAnalyzer()
    analyzer.run_full_analysis() 