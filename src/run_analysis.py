"""
Script principal pour exécuter l'analyse de données des tweets.
"""

import logging
from pathlib import Path
from src.analysis.exploratory_analysis import TweetAnalyzer

def setup_logging():
    """Configure le logging pour l'analyse."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Fonction principale pour exécuter l'analyse."""
    logger = setup_logging()
    logger.info("Démarrage de l'analyse des données...")
    
    try:
        # Création des répertoires nécessaires
        for dir_name in ['results/visualizations', 'data/processed']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        # Exécution de l'analyse
        analyzer = TweetAnalyzer()
        analyzer.run_full_analysis()
        
        logger.info("Analyse terminée avec succès!")
        logger.info("Les visualisations sont disponibles dans le dossier 'results/visualizations'")
        logger.info("Le rapport sommaire est disponible dans 'results/visualizations/analysis_summary.json'")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue pendant l'analyse : {str(e)}")
        raise

if __name__ == '__main__':
    main() 