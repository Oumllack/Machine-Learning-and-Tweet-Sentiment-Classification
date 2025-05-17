# Analyse de Sentiments Twitter 🐦

Ce projet vise à classifier automatiquement les tweets en sentiments positifs, négatifs ou neutres en utilisant des techniques de machine learning.

## 📋 Description

Le projet comprend :
- Collecte et prétraitement des tweets
- Analyse exploratoire des données
- Construction de modèles de classification
- Visualisation des résultats

## 🚀 Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd twitter-sentiment-classification
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/MacOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Télécharger les ressources NLTK nécessaires :
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. Télécharger le modèle spaCy :
```bash
python -m spacy download fr_core_news_md
```

## 📁 Structure du Projet

```
twitter-sentiment-classification/
├── data/               # Données brutes et traitées
├── notebooks/          # Notebooks Jupyter pour l'analyse
├── src/               # Code source
│   ├── data/          # Scripts de prétraitement
│   ├── models/        # Modèles de classification
│   └── utils/         # Utilitaires
├── requirements.txt    # Dépendances Python
└── README.md          # Documentation
```

## 🎯 Utilisation

1. Préparation des données :
```bash
python src/data/preprocess.py
```

2. Entraînement du modèle :
```bash
python src/models/train.py
```

3. Évaluation :
```bash
python src/models/evaluate.py
```

## 📊 Résultats

Les résultats incluent :
- Matrice de confusion
- Métriques de performance (précision, rappel, F1-score)
- Visualisations des sentiments
- Nuages de mots

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/Amelioration`)
3. Commit vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push sur la branche (`git push origin feature/Amelioration`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 