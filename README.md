# Gestion des Absences — ISIMS Sfax

Application web de gestion des absences étudiantes développée dans le cadre d'un stage à l'ISIMS Sfax (Institut Supérieur d'Informatique et de Multimédia de Sfax).

## 📋 Présentation

L'application permet aux enseignants et administrateurs de gérer les absences des étudiants : saisie, consultation, justification, génération de rapports, avec une gestion complète multi-années universitaires.

**Fonctionnalités principales :**
- Gestion des utilisateurs (admin / enseignant) avec authentification sécurisée et 2FA
- Gestion des étudiants, classes, matières, programmes et affectations, sur plusieurs années universitaires
- Saisie et suivi des absences, avec durée automatique selon la matière
- Système de justificatif d'absence en ligne (lien sécurisé envoyé par email, upload de preuve, validation admin)
- Escalade automatique en cas d'absences répétées non justifiées
- Tableau de bord avec statistiques et graphiques
- Export des rapports au format PDF et Excel
- Chatbot d'assistance intégré

## 🛠️ Stack technique

- **Backend** : Python 3.10+, Flask, SQLAlchemy
- **Base de données** : SQLite
- **Frontend** : HTML5, CSS3, Bootstrap 5, Jinja2
- **Emails** : Brevo (API transactionnelle)
- **Sécurité** : Flask-Login, Flask-WTF (CSRF), Flask-Limiter (rate limiting), authentification à deux facteurs

## 🚀 Installation

### Prérequis
- Python 3.10 ou supérieur
- pip

### Étapes

1. **Cloner le dépôt**
```bash
git clone <url-du-depot>
cd projet_absences
```

2. **Créer et activer un environnement virtuel**
```bash
python -m venv venv
```
Sur Windows :
```bash
venv\Scripts\activate
```
Sur macOS/Linux :
```bash
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**

Créer un fichier `.env` à la racine du projet avec le contenu suivant :
```
SECRET_KEY=votre_cle_secrete_generee
GROQ_API_KEY=votre_cle_api_groq
BREVO_API_KEY=votre_cle_api_brevo
BREVO_EMAIL_EXPEDITEUR=votre_adresse_expediteur_validee
```

Pour générer une `SECRET_KEY` aléatoire :
```python
import secrets
print(secrets.token_hex(16))
```

5. **Lancer l'application**
```bash
python app.py
```

L'application est accessible à l'adresse : `http://127.0.0.1:5000`

## 📁 Structure du projet

```
projet_absences/
├── app.py                  # Point d'entrée Flask, toutes les routes
├── models.py                # Modèles SQLAlchemy
├── requirements.txt          # Dépendances Python
├── .env                      # Variables d'environnement (non versionné)
├── templates/                 # Templates Jinja2
├── static/
│   └── css/style.css          # Feuille de style principale
├── uploads_justificatifs/      # Fichiers uploadés (non versionné)
├── backups/                    # Sauvegardes automatiques de la base (non versionné)
└── instance/
    └── absences.db              # Base de données SQLite
```

## 🔐 Compte de test

Après le premier lancement, un compte administrateur peut être créé via la route `/register` (nécessite d'être déjà connecté en tant qu'admin, ou de créer le premier compte directement en base).

## 📖 Documentation complémentaire

Pour plus de détails sur l'architecture, les choix de conception et le manuel d'utilisation, se référer au rapport de stage.

## 👤 Auteur

Projet réalisé dans le cadre d'un stage de fin d'études — 2ème année cycle ingénieur.
