import re
import logging
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

class TweetPreprocessor:
    def __init__(self, language='english'):
        """Initialise le prétraitement des tweets."""
        self.language = language
        logger.info(f"Initialized TweetPreprocessor for {language} language")
        
    def preprocess_text(self, text: str) -> str:
        """
        Prétraite un tweet.
        
        Args:
            text (str): Texte du tweet
            
        Returns:
            str: Texte prétraité
        """
        if not isinstance(text, str):
            return ""
            
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer les mentions @
        text = re.sub(r'@\w+', '', text)
        
        # Supprimer les hashtags (garder le texte)
        text = re.sub(r'#', '', text)
        
        # Supprimer les caractères spéciaux et la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
        """
        Prétraite le dataset entier.
        
        Args:
            df (pd.DataFrame): DataFrame d'entrée
            text_column (str): Nom de la colonne de texte
            label_column (str): Nom de la colonne de label
            
        Returns:
            pd.DataFrame: DataFrame prétraité
        """
        logger.info("Starting dataset preprocessing...")
        
        # Copie pour éviter de modifier l'original
        df_processed = df.copy()
        
        # Prétraitement par lots pour gérer la mémoire
        logger.info("Processing tweets in batches...")
        batch_size = 100000  # Traiter 100K tweets à la fois
        total_tweets = len(df_processed)
        num_batches = (total_tweets + batch_size - 1) // batch_size
        
        processed_texts = []
        empty_count = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_tweets)
            logger.info(f"Processing batch {i+1}/{num_batches} ({start_idx}-{end_idx})")
            
            # Prétraiter le lot actuel
            batch_texts = df_processed.iloc[start_idx:end_idx][text_column]
            batch_processed = []
            
            for text in tqdm(batch_texts, desc=f"Batch {i+1}"):
                processed = self.preprocess_text(text)
                if not processed.strip():
                    empty_count += 1
                batch_processed.append(processed)
            
            processed_texts.extend(batch_processed)
            logger.info(f"Completed batch {i+1}")
        
        # Ajouter les textes prétraités
        df_processed['processed_text'] = processed_texts
        
        # Supprimer les tweets vides
        initial_len = len(df_processed)
        df_processed = df_processed[df_processed['processed_text'].str.strip() != '']
        final_len = len(df_processed)
        
        logger.info(f"Preprocessing completed:")
        logger.info(f"- Total tweets processed: {initial_len}")
        logger.info(f"- Empty tweets removed: {empty_count}")
        logger.info(f"- Remaining tweets: {final_len}")
        
        return df_processed 