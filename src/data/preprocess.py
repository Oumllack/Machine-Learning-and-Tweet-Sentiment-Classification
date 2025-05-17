import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TweetPreprocessor:
    def __init__(self, language='french'):
        """
        Initialise le préprocesseur de tweets.
        
        Args:
            language (str): Langue des tweets ('french' ou 'english')
        """
        self.language = language
        self.stop_words = set(stopwords.words('french' if language == 'french' else 'english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('fr_core_news_md' if language == 'french' else 'en_core_web_md')
        
    def remove_urls(self, text):
        """Supprime les URLs du texte."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_emojis(self, text):
        """Supprime les emojis du texte."""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    
    def remove_mentions_and_hashtags(self, text):
        """Supprime les mentions (@) et les hashtags (#)."""
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text
    
    def remove_special_chars(self, text):
        """Supprime les caractères spéciaux tout en gardant les accents."""
        return re.sub(r'[^\w\s]', ' ', text)
    
    def lemmatize_text(self, text):
        """Lemmatise le texte en utilisant spaCy."""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def preprocess_text(self, text):
        """
        Applique toutes les étapes de prétraitement sur un texte.
        
        Args:
            text (str): Texte à prétraiter
            
        Returns:
            str: Texte prétraité
        """
        if not isinstance(text, str):
            return ''
            
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs, emojis, mentions et hashtags
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.remove_mentions_and_hashtags(text)
        text = self.remove_special_chars(text)
        
        # Lemmatisation
        text = self.lemmatize_text(text)
        
        # Suppression des stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def preprocess_dataset(self, df, text_column, target_column=None):
        """
        Prétraite un dataset complet de tweets.
        
        Args:
            df (pandas.DataFrame): DataFrame contenant les tweets
            text_column (str): Nom de la colonne contenant le texte des tweets
            target_column (str, optional): Nom de la colonne cible
            
        Returns:
            pandas.DataFrame: DataFrame prétraité
        """
        logger.info("Début du prétraitement du dataset...")
        
        # Copie du DataFrame pour éviter de modifier l'original
        df_processed = df.copy()
        
        # Prétraitement du texte
        tqdm.pandas(desc="Prétraitement des tweets")
        df_processed['processed_text'] = df_processed[text_column].progress_apply(self.preprocess_text)
        
        # Suppression des lignes avec texte vide après prétraitement
        initial_len = len(df_processed)
        df_processed = df_processed[df_processed['processed_text'].str.strip() != '']
        final_len = len(df_processed)
        
        logger.info(f"Prétraitement terminé. {initial_len - final_len} tweets vides supprimés.")
        
        return df_processed

def main():
    """Fonction principale pour tester le prétraitement."""
    # Exemple d'utilisation
    sample_tweets = pd.DataFrame({
        'text': [
            "J'adore ce nouveau film ! 🎬 #cinema @cinephile",
            "Le service client est horrible... https://example.com",
            "RT @user: Belle journée aujourd'hui ! ☀️"
        ],
        'sentiment': [1, 0, 1]  # 1: positif, 0: négatif
    })
    
    preprocessor = TweetPreprocessor(language='french')
    processed_df = preprocessor.preprocess_dataset(sample_tweets, 'text', 'sentiment')
    
    print("\nExemple de tweets prétraités :")
    print(processed_df[['text', 'processed_text']].head())

if __name__ == "__main__":
    main() 