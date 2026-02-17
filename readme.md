# Ticket Classification MLOps

# Industrialisation d'un Pipeline NLP de Classification de Tickets Support avec MLOps

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Supported-326CE5.svg)](https://kubernetes.io/)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-green.svg)](https://ml-ops.org/)

---

## Contexte du Projet

Ce projet a été réalisé dans le cadre d'une mission en entreprise IT disposant d'un historique d'emails de support client.

Chaque ticket contient :
- **Champs textuels** : `subject`, `body`, `answer`
- **Métadonnées métier** : priorité, type, queue, langue, etc.

### Objectif

Industrialiser un pipeline batch NLP permettant de :

1. Traiter et comprendre le contenu des emails support
2. Générer des représentations sémantiques (embeddings) avec un modèle Hugging Face
3. Entraîner un modèle de classification supervisée pour prédire le type de ticket
4. Stocker les embeddings dans une base vectorielle ChromaDB
5. Surveiller la qualité du modèle et la dérive des données avec Evidently AI
6. Superviser l'infrastructure avec Prometheus et Grafana

> **Note** : L'ensemble du projet est exécuté dans un environnement containerisé (Docker & Kubernetes) **sans exposition d'API**.

---

## Structure du Projet


```
├── 📁 .github
│   └── 📁 workflows
│       └── ⚙️ ml-pipeline.yml
├── 📁 Nootbooks
│   └── 📄 eda.ipynb
├── 📁 data
├── 📁 k8s
│   ├── ⚙️ chromadb.yaml
│   └── ⚙️ ml.yaml
├── 📁 models
│   ├── 📁 artifacts
│   └── 📁 trained
├── 📁 monitoring
│   ├── 📁 grafana
│   │   └── 📁 dashboards
│   └── 📁 prometheus
├── 📁 src
│   ├── 📁 data
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 cleaning.py
│   │   ├── 📄 dataset.csv
│   │   ├── 🐍 encoding.py
│   │   └── 🐍 load_data.py
│   ├── 📁 features
│   │   └── 🐍 embedding_generator.py
│   ├── 📁 models
│   │   ├── 🐍 evaluate.py
│   │   ├── 🐍 split.py
│   │   └── 🐍 train.py
│   ├── 📁 monitoring
│   │   ├── 🐍 drift_detection.py
│   │   └── 🐍 evendently_report.py
│   ├── 📁 vectors_store
│   │   └── 🐍 chromadb_client.py
│   └── 🐍 __init__.py
├── ⚙️ .dockerignore
├── ⚙️ .env.example
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── ⚙️ docker-compose.yml
├── ⚙️ prometheus.yml
├── 📝 readme.md
└── 📄 requirements.txt
```
---

## Étapes du Pipeline

### 1. Analyse Exploratoire & Préparation NLP

- Analyse des types de tickets, longueur des emails
- Fusion des champs textuels (`subject + body`)
- Nettoyage NLP :
  - Conversion en minuscules
  - Suppression de la ponctuation
  - Tokenisation
  - Suppression des stopwords selon la langue

### 2. Génération d'Embeddings

- Sélection d'un modèle pré-entraîné Hugging Face (`all-MiniLM-L6-v2`)
- Encodage des textes nettoyés en vecteurs
- Normalisation des vecteurs
- Stockage dans ChromaDB

### 3. Entraînement du Modèle de Classification

- Séparation train/test
- Entraînement avec scikit-learn (ex: RandomForest ou Logistic Regression)
- Évaluation : précision, recall, F1-score

### 4. Monitoring ML avec Evidently AI

- Définition d'un jeu de référence (baseline)
- Suivi de **data drift** et **prediction drift**
- Génération de rapports HTML interactifs

### 5. Conteneurisation et Orchestration

- Dockerisation du pipeline NLP & ML
- Déploiement batch sur Kubernetes (Minikube)
- CI/CD avec GitHub Actions (lint + build Docker)

### 6. Monitoring Infrastructure avec Prometheus & Grafana

- **Node Exporter** : métriques CPU/RAM/disque
- **cAdvisor** : consommation des containers Docker
- Dashboards Grafana configurés pour visualisation

---

## 🛠️ Dépendances Principales

- Python >= 3.10
- pandas, numpy, scikit-learn, nltk
- langchain_community.embeddings
- ChromaDB
- Docker & Kubernetes
- Prometheus & Grafana
- Evidently AI

---

## Instructions pour Exécuter le Pipeline

### 1. Installer l'Environnement

```bash
python -m venv env
source env/bin/activate  # Sur Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Prétraitement des Données

```python
from src.data.load_data import load_data
from src.data.cleaning import cleaning
from src.data.encoding import encoding

df = load_data()
df_clean = cleaning(df)
df_encoded = encoding(df_clean)
```

### 3. Génération et Stockage des Embeddings

```python
from src.features.embedding_generator import generate_embeddings

generate_embeddings(df_encoded)
```

### 4. Exécution du Modèle de Classification

```python
from src.models.classifier import train_model, evaluate_model

model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)
```

### 5. Monitoring ML

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Génération de rapports Evidently AI pour data/prediction drift
report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("drift_report.html")
```

### 6. Exécution Containerisée

```bash
# Build Docker image
docker build -t ticket-nlp-pipeline ./docker

# Déploiement sur Kubernetes
kubectl apply -f k8s/
```

### 7. Monitoring Infrastructure

```bash
cd monitoring
docker-compose up -d
```

Accès aux interfaces :
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3000 (admin/admin)

---

## Livrables

- ✅ Scripts de preprocessing NLP
- ✅ Embeddings stockés dans ChromaDB
- ✅ Modèle de classification entraîné
- ✅ Rapports Evidently AI
- ✅ Images Docker & manifests Kubernetes
- ✅ Rapport technique final

---

## Critères de Performance

| Critère | Description |
|---------|-------------|
| **Qualité NLP** | Nettoyage efficace des textes, tokenisation adaptée |
| **Embeddings** | Cohérence des vecteurs et indexation ChromaDB optimale |
| **Classification** | Précision > 85%, F1-score équilibré |
| **Monitoring ML** | Détection proactive de drift avec Evidently AI |
| **Infrastructure** | Dashboards Prometheus/Grafana opérationnels |
| **Reproductibilité** | Pipeline complètement automatisé via Docker/Kubernetes |

---

## Remarques

- Le pipeline est conçu pour être **batch**, non exposé en API
- Toutes les étapes sont supervisées via MLOps pour garantir stabilité et maintenance continue
- L'orchestration Kubernetes permet une scalabilité horizontale
- Les métriques de monitoring permettent une détection précoce des anomalies

---


