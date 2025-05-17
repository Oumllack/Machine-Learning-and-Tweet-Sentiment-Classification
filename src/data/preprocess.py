import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import contractions
import emoji
from typing import List, Dict, Union
from multiprocessing import Pool, cpu_count

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TweetPreprocessor:
    def __init__(self, language='english'):
        """Initialize preprocessor with language-specific settings."""
        self.language = language.lower()
        self.stop_words = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        self.vader = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}') if resource in ('stopwords', 'wordnet') else nltk.data.find(f'taggers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
        
        logger.info(f"Initialized TweetPreprocessor for {self.language} language")
        
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text (e.g., "don't" -> "do not")."""
        return contractions.fix(text)
    
    def get_wordnet_pos(self, word: str) -> str:
        """Get WordNet POS tag using spaCy."""
        # Utiliser spaCy pour le POS tagging
        doc = self.nlp(word)
        if not doc:
            return wordnet.NOUN
        
        pos = doc[0].pos_
        tag_dict = {
            "ADJ": wordnet.ADJ,
            "NOUN": wordnet.NOUN,
            "VERB": wordnet.VERB,
            "ADV": wordnet.ADV
        }
        return tag_dict.get(pos, wordnet.NOUN)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+')
        return url_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis and convert them to text description."""
        return emoji.replace_emoji(text, replace='')
    
    def remove_mentions_and_hashtags(self, text: str) -> str:
        """Remove mentions and hashtags, but keep the text content."""
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove # but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters while preserving important punctuation."""
        # Keep important punctuation for sentiment
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using VADER and TextBlob."""
        vader_scores = self.vader.polarity_scores(text)
        blob = TextBlob(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy and WordNet."""
        doc = self.nlp(text)
        lemmatized = []
        
        for token in doc:
            if not token.is_stop and not token.is_punct:
                # Utiliser spaCy pour le POS tagging
                pos = token.pos_
                if pos in ['ADJ', 'NOUN', 'VERB', 'ADV']:
                    lemmatized.append(token.lemma_)
                else:
                    lemmatized.append(token.text.lower())
        
        return ' '.join(lemmatized)
    
    def preprocess_text(self, text: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Apply all preprocessing steps and extract features.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            dict: Dictionary containing processed text and features
        """
        if not isinstance(text, str):
            return {'processed_text': '', 'features': {}}
        
        # Basic cleaning
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.remove_urls(text)
        text = self.remove_mentions_and_hashtags(text)
        
        # Extract sentiment features before removing emojis
        features = self.get_sentiment_features(text)
        
        # Continue cleaning
        text = self.remove_emojis(text)
        text = self.remove_special_chars(text)
        text = self.lemmatize_text(text)
        
        # Remove stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return {
            'processed_text': ' '.join(words),
            'features': features
        }
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str, target_column: str = None) -> pd.DataFrame:
        """
        Preprocess entire dataset with advanced features.
        
        Args:
            df (pandas.DataFrame): DataFrame containing tweets
            text_column (str): Name of the text column
            target_column (str, optional): Name of the target column
            
        Returns:
            pandas.DataFrame: Preprocessed DataFrame with additional features
        """
        logger.info("Starting dataset preprocessing...")
        
        df_processed = df.copy()
        
        # Traitement séquentiel pour debug
        logger.info("Processing tweets sequentially...")
        processed_results = []
        
        # Traiter par lots de 1000 tweets pour suivre la progression
        batch_size = 1000
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
        
        for i in range(0, len(df), batch_size):
            batch = df_processed[text_column].iloc[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")
            
            # Traiter chaque tweet du lot
            batch_results = []
            for text in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
                try:
                    result = self.preprocess_text(text)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing tweet: {str(e)}")
                    batch_results.append({'processed_text': '', 'features': {}})
            
            processed_results.extend(batch_results)
            logger.info(f"Completed batch {i//batch_size + 1}")
        
        # Convertir les résultats en DataFrame
        processed_df = pd.DataFrame(processed_results)
        
        # Extraire le texte traité et les features
        df_processed['processed_text'] = processed_df['processed_text']
        features_df = pd.DataFrame(processed_df['features'].tolist())
        
        # Combiner avec le DataFrame original
        df_processed = pd.concat([df_processed, features_df], axis=1)
        
        # Supprimer les textes vides
        initial_len = len(df_processed)
        df_processed = df_processed[df_processed['processed_text'].str.strip() != '']
        final_len = len(df_processed)
        
        logger.info(f"Preprocessing completed. {initial_len - final_len} empty tweets removed.")
        logger.info(f"Added {len(features_df.columns)} sentiment features.")
        
        return df_processed

def main():
    """Test the preprocessing with example tweets."""
    sample_tweets = pd.DataFrame({
        'text': [
            "I love this new movie! 🎬 #cinema @moviebuff",
            "Customer service is terrible... https://example.com",
            "RT @user: Beautiful day today! ☀️",
            "Don't you think this is amazing?",
            "The product is good, but the price is too high :("
        ],
        'sentiment': [1, 0, 1, 1, 0]
    })
    
    preprocessor = TweetPreprocessor(language='english')
    processed_df = preprocessor.preprocess_dataset(sample_tweets, 'text', 'sentiment')
    
    print("\nExample of preprocessed tweets with features:")
    print(processed_df[['text', 'processed_text', 'vader_compound', 'textblob_polarity']].head())

if __name__ == "__main__":
    main() 