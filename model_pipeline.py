"""
model_pipeline.py — Pipeline ML modulaire (base Adult UCI, objectif C).

Objectif : prédire `has_capital_gain` (1 = la personne a un gain en capital).

Fonctions principales (celles demandées par l'Atelier 2) :
    prepare_data()   : charger, nettoyer, transformer les données
    train_model()    : entraîner un modèle (fit)
    evaluate_model() : évaluer les performances sur le test
    save_model()     : sauvegarder le modèle entraîné (joblib)
    load_model()     : recharger un modèle sauvegardé

Fonctions complémentaires : compare_models(), tune_xgboost(), find_best_threshold(),
predict_new().
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42  # graine fixe : résultats reproductibles
TARGET = "has_capital_gain"

COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

MODEL_NAMES = ["logreg", "rf", "adaboost", "xgboost", "voting"]
IMBALANCE_METHODS = ["class_weight", "smote", "undersample", "none"]

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


# --------------------------------------------------------------------------- #
# Transformateur personnalisé (doit rester dans ce fichier : joblib en a besoin
# pour recharger le préprocesseur)
# --------------------------------------------------------------------------- #
class Ecreteur(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Plafonne (clip) les valeurs extrêmes. Les bornes sont apprises au fit (donc sur le train).

    methode = "iqr"        -> [Q1 - 1.5*IQR ; Q3 + 1.5*IQR]
    methode = "percentile" -> [percentile 1 % ; percentile 99 %]
    """

    def __init__(self, methode="iqr"):
        self.methode = methode

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.methode == "iqr":
            q1, q3 = X.quantile(0.25), X.quantile(0.75)
            iqr = q3 - q1
            self.bas_, self.haut_ = (q1 - 1.5 * iqr).to_numpy(), (q3 + 1.5 * iqr).to_numpy()
        else:
            self.bas_, self.haut_ = X.quantile(0.01).to_numpy(), X.quantile(0.99).to_numpy()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.bas_, self.haut_)


# --------------------------------------------------------------------------- #
# 1. Préparation des données
# --------------------------------------------------------------------------- #
def load_data(path="adult.data"):
    """Charge adult.data (le "?" devient NaN). Si le fichier local n'existe pas, essaie l'URL UCI."""
    source = path if os.path.exists(path) else URL
    df = pd.read_csv(source, names=COLS, na_values="?", skipinitialspace=True)
    return _normaliser_colonnes(df)


def _normaliser_colonnes(df):
    df = df.copy()
    df.columns = df.columns.str.replace(r"[-.]", "_", regex=True)
    return df


def clean_data(df):
    """Nettoyage sans calcul statistique : doublons + colonnes redondantes (education, fnlwgt)."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns=["education", "fnlwgt"], errors="ignore")


def engineer_features(df):
    """Feature engineering ligne par ligne (sans risque avant le découpage train/test).

    - crée la cible `has_capital_gain` si `capital_gain` est présente ;
    - crée is_full_time, is_us ; encode sex et income en 0/1 ;
    - retire capital_gain / capital_loss (fuite de données) et native_country.
    """
    df = df.copy()
    if "capital_gain" in df.columns:
        df[TARGET] = (df["capital_gain"] > 0).astype(int)
    df["is_full_time"] = (df["hours_per_week"] >= 40).astype(int)
    df["is_us"] = np.where(df["native_country"].isna(), np.nan, (df["native_country"] == "United-States").astype(int))
    df["sex"] = (df["sex"] == "Male").astype(int)
    df["income"] = (df["income"] == ">50K").astype(int)
    return df.drop(columns=["native_country", "capital_gain", "capital_loss"], errors="ignore")


def build_preprocessor():
    """ColumnTransformer : imputation (mode), plafonnement, normalisation, one-hot."""
    age_cols = ["age"]
    hours_cols = ["hours_per_week"]
    edu_cols = ["education_num"]
    bin_cols = ["sex", "income", "is_full_time", "is_us"]
    cat_cols = ["workclass", "marital_status", "occupation", "relationship", "race"]

    return ColumnTransformer(
        transformers=[
            ("age", Pipeline([("clip", Ecreteur("iqr")), ("scale", StandardScaler())]), age_cols),
            (
                "hours",
                Pipeline([("clip", Ecreteur("percentile")), ("scale", StandardScaler())]),
                hours_cols,
            ),
            ("edu", StandardScaler(), edu_cols),
            ("bin", SimpleImputer(strategy="most_frequent"), bin_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(data_path="adult.data", test_size=0.2, save_dir=None):
    """Charger et prétraiter les données.

    Étapes : chargement -> nettoyage -> feature engineering -> split stratifié ->
    transformation (apprise sur le TRAIN uniquement, appliquée au train et au test).

    Args:
        data_path: chemin du fichier adult.data.
        test_size: part du jeu de test (0.2 = 20 %).
        save_dir: si renseigné, sauvegarde train.csv, test.csv et preprocess.joblib dedans.

    Returns:
        X_train_p, X_test_p, y_train, y_test, preprocess
    """
    df = engineer_features(clean_data(load_data(data_path)))
    if TARGET not in df.columns:
        raise ValueError("Colonne cible introuvable : le fichier doit contenir capital-gain.")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    preprocess = build_preprocessor()
    cols = None
    X_train_p = pd.DataFrame(
        preprocess.fit_transform(X_train),
        columns=(cols := preprocess.get_feature_names_out()),
        index=X_train.index,
    )
    X_test_p = pd.DataFrame(preprocess.transform(X_test), columns=cols, index=X_test.index)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        X_train_p.assign(**{TARGET: y_train}).to_csv(os.path.join(save_dir, "train.csv"), index=False)
        X_test_p.assign(**{TARGET: y_test}).to_csv(os.path.join(save_dir, "test.csv"), index=False)
        joblib.dump(preprocess, os.path.join(save_dir, "preprocess.joblib"))

    return X_train_p, X_test_p, y_train, y_test, preprocess


def load_prepared_data(save_dir):
    """Recharge les fichiers créés par prepare_data(save_dir=...)."""
    train = pd.read_csv(os.path.join(save_dir, "train.csv"))
    test = pd.read_csv(os.path.join(save_dir, "test.csv"))
    return (train.drop(columns=TARGET), test.drop(columns=TARGET), train[TARGET], test[TARGET])


# --------------------------------------------------------------------------- #
# 2. Entraînement
# --------------------------------------------------------------------------- #
def build_model(model_name, y_train, imbalance="class_weight", xgb_params=None):
    """Construit (sans entraîner) un modèle.

    Args:
        model_name: logreg | rf | adaboost | xgboost | voting
        imbalance: class_weight | smote | undersample | none (rééquilibrage, TRAIN seulement)
        xgb_params: hyperparamètres XGBoost (ex. issus de tune_xgboost)
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Modèle inconnu '{model_name}'. Choix : {MODEL_NAMES}")
    if imbalance not in IMBALANCE_METHODS:
        raise ValueError(f"Méthode inconnue '{imbalance}'. Choix : {IMBALANCE_METHODS}")

    poids = imbalance == "class_weight"
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    lr = LogisticRegression(max_iter=3000, class_weight="balanced" if poids else None)
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced" if poids else None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced" if poids else None),
        n_estimators=200,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio if poids else 1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if xgb_params:
        xgb.set_params(**xgb_params)
    voting = VotingClassifier(
        estimators=[("lr", clone(lr)), ("rf", clone(rf)), ("ada", clone(ada)), ("xgb", clone(xgb))],
        voting="soft",
    )

    model = {"logreg": lr, "rf": rf, "adaboost": ada, "xgboost": xgb, "voting": voting}[model_name]

    if imbalance == "smote":
        return ImbPipeline([("sampler", SMOTE(random_state=RANDOM_STATE)), ("model", model)])
    if imbalance == "undersample":
        return ImbPipeline([("sampler", RandomUnderSampler(random_state=RANDOM_STATE)), ("model", model)])
    return model


def train_model(X_train, y_train, model_name="xgboost", imbalance="class_weight", xgb_params=None):
    """Entraîner le modèle (fit) sur le train et le retourner."""
    model = build_model(model_name, y_train, imbalance, xgb_params)
    model.fit(X_train, y_train)
    return model


def compare_models(X_train, y_train, imbalance="class_weight", n_splits=5):
    """Compare les 5 modèles par validation croisée stratifiée (sur le train). Trié par PR-AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    lignes = []
    for name in MODEL_NAMES:
        res = cross_validate(
            build_model(name, y_train, imbalance),
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
        )
        lignes.append(
            {
                "modele": name,
                **{k: res[f"test_{k}"].mean() for k in SCORING},
                "pr_auc_std": res["test_pr_auc"].std(),
            }
        )
    return pd.DataFrame(lignes).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def tune_xgboost(X_train, y_train, n_iter=20):
    """RandomizedSearchCV sur XGBoost (score = PR-AUC, sur le train uniquement).

    Returns: (meilleurs_parametres, meilleur_score)
    """
    param_dist = {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.5, 1],
        "reg_lambda": [1, 5, 10],
    }
    search = RandomizedSearchCV(
        build_model("xgboost", y_train),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


def find_best_threshold(model, X_train, y_train, n_splits=5):
    """Seuil qui maximise le F1, calculé sur des prédictions « hors pli » du train (le test n'est pas touché)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    prec, rapp, seuils = precision_recall_curve(y_train, proba)
    f1 = 2 * prec[:-1] * rapp[:-1] / (prec[:-1] + rapp[:-1] + 1e-12)
    return float(seuils[np.argmax(f1)])


# --------------------------------------------------------------------------- #
# 3. Évaluation
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test, threshold=0.5, plots_dir=None):
    """Évaluer les performances sur le test.

    Métriques adaptées aux classes déséquilibrées (PR-AUC, rappel, précision, F1).

    Args:
        threshold: seuil de décision (0,5 par défaut).
        plots_dir: si renseigné, enregistre la matrice de confusion et la courbe précision-rappel.

    Returns:
        dict de métriques.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "seuil": round(float(threshold), 3),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "rappel": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "taux_de_base": float(np.mean(y_test)),
    }
    metrics["matrice_confusion"] = confusion_matrix(y_test, pred).tolist()

    if plots_dir:
        _sauver_graphiques(y_test, proba, pred, plots_dir)
    return metrics


def _sauver_graphiques(y_test, proba, pred, plots_dir):
    import matplotlib

    matplotlib.use("Agg")  # pas d'écran nécessaire (WSL / serveur)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Non", "Oui"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Matrice de confusion (test)")
    p, r, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(r, p, label=f"PR-AUC = {average_precision_score(y_test, proba):.2f}")
    axes[1].axhline(np.mean(y_test), color="k", ls="--", label="Hasard")
    axes[1].set(title="Courbe précision-rappel (test)", xlabel="Rappel", ylabel="Précision")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "evaluation.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Sauvegarde / chargement
# --------------------------------------------------------------------------- #
def save_model(model, path="artifacts/model.joblib", threshold=0.5, name="xgboost", imbalance="class_weight"):
    """Sauvegarde le modèle entraîné + son seuil (joblib)."""
    dossier = os.path.dirname(path)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    joblib.dump({"modele": model, "seuil": threshold, "nom": name, "desequilibre": imbalance}, path)
    return path


def load_model(path="artifacts/model.joblib"):
    """Charge un modèle sauvegardé. Retourne un dict : {modele, seuil, nom, desequilibre}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}. Lance d'abord l'étape 'train'.")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# 5. Prédiction sur de nouvelles données brutes
# --------------------------------------------------------------------------- #
def predict_new(df_raw, preprocess, bundle):
    """Prédit has_capital_gain pour de nouvelles lignes BRUTES (mêmes colonnes que adult.data, sans cible obligatoire).

    Returns: DataFrame avec `probabilite` et `prediction` (0/1, selon le seuil sauvegardé).
    """
    df = engineer_features(_normaliser_colonnes(df_raw))
    X = pd.DataFrame(preprocess.transform(df), columns=preprocess.get_feature_names_out(), index=df.index)
    proba = bundle["modele"].predict_proba(X)[:, 1]
    return pd.DataFrame({"probabilite": proba, "prediction": (proba >= bundle["seuil"]).astype(int)}, index=df.index)
