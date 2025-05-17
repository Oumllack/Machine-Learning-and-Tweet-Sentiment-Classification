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
        """Analyze temporal patterns of tweets."""
        logger.info("Analyzing temporal patterns...")
        
        # Distribution of tweets by day
        plt.figure(figsize=(15, 6))
        self.df.groupby(self.df['date'].dt.date).size().plot(kind='line')
        plt.title('Tweet Distribution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tweets_over_time.png')
        plt.close()
        
        # Distribution of sentiments by day
        plt.figure(figsize=(15, 6))
        daily_sentiment = self.df.groupby([self.df['date'].dt.date, 'sentiment']).size().unstack()
        daily_sentiment.plot(kind='line')
        plt.title('Sentiment Distribution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.legend(['Negative', 'Positive'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sentiment_over_time.png')
        plt.close()
        
    def analyze_text_length(self):
        """Analyze tweet length distribution."""
        logger.info("Analyzing tweet length distribution...")
        
        self.df['tweet_length'] = self.df['text'].str.len()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='tweet_length', hue='sentiment', bins=50)
        plt.title('Tweet Length Distribution by Sentiment')
        plt.xlabel('Tweet Length (characters)')
        plt.ylabel('Count')
        plt.legend(['Negative', 'Positive'])
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tweet_length_distribution.png')
        plt.close()
        
        # Statistiques de longueur par sentiment
        length_stats = self.df.groupby('sentiment')['tweet_length'].agg(['mean', 'std', 'min', 'max'])
        logger.info("\nStatistiques de longueur par sentiment :\n" + str(length_stats))
        
    def generate_wordclouds(self):
        """Generate word clouds for each sentiment."""
        logger.info("Generating word clouds...")
        
        # Positive sentiment word cloud
        positive_text = ' '.join(self.df[self.df['sentiment'] == 1]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud - Positive Sentiment')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'wordcloud_positive.png')
        plt.close()
        
        # Negative sentiment word cloud
        negative_text = ' '.join(self.df[self.df['sentiment'] == 0]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud - Negative Sentiment')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'wordcloud_negative.png')
        plt.close()
        
    def analyze_common_words(self):
        """Analyze most common words by sentiment."""
        logger.info("Analyzing common words...")
        
        # Positive tweets
        positive_words = ' '.join(self.df[self.df['sentiment'] == 1]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Most Common Words in Positive Tweets')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'common_words_positive.png')
        plt.close()
        
        # Negative tweets
        negative_words = ' '.join(self.df[self.df['sentiment'] == 0]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_words)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Most Common Words in Negative Tweets')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'common_words_negative.png')
        plt.close()
        
    def analyze_user_activity(self):
        """Analyze user activity patterns."""
        logger.info("Analyzing user activity...")
        
        # Number of tweets per user
        user_activity = self.df['user'].value_counts()
        
        plt.figure(figsize=(12, 6))
        user_activity.head(20).plot(kind='bar')
        plt.title('Top 20 Most Active Users')
        plt.xlabel('User')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'user_activity.png')
        plt.close()
        
        # Distribution of tweets per user
        plt.figure(figsize=(12, 6))
        sns.histplot(user_activity, bins=50)
        plt.title('Distribution of Tweets per User')
        plt.xlabel('Number of Tweets')
        plt.ylabel('Number of Users')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'user_activity_distribution.png')
        plt.close()
        
    def analyze_sentiment_patterns(self):
        """Analyze sentiment distribution."""
        logger.info("Analyzing sentiment distribution...")
        
        plt.figure(figsize=(10, 6))
        self.df['sentiment'].value_counts().plot(kind='bar')
        plt.title('Overall Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sentiment_distribution.png')
        plt.close()
        
    def analyze_user_sentiment(self):
        """Analyze user sentiment patterns."""
        logger.info("Analyzing user sentiment patterns...")
        
        # Average sentiment by user
        user_sentiment = self.df.groupby('user')['sentiment'].mean()
        
        plt.figure(figsize=(12, 6))
        user_sentiment.head(20).plot(kind='bar')
        plt.title('Average Sentiment for Top 20 Users')
        plt.xlabel('User')
        plt.ylabel('Average Sentiment (0=Negative, 1=Positive)')
        plt.xticks(rotation=45)
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
        self.analyze_user_sentiment()
        self.generate_summary_report()
        
        logger.info("Analyse complète terminée. Les visualisations sont disponibles dans le dossier 'results/visualizations'")

if __name__ == '__main__':
    analyzer = TweetAnalyzer()
    analyzer.run_full_analysis() 