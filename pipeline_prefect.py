"""pipeline_prefect.py : orchestration du pipeline ML avec Prefect."""

import argparse
import os
import subprocess  # nosec B404
import sys
from types import SimpleNamespace

from prefect import flow, get_run_logger, task

import main as m
from model_pipeline import load_model, save_model  # noqa: F401

# Étapes qualité/sécurité limitées à nos 3 fichiers (RQ de l'atelier)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def make_args(**kwargs):
    """Construit l'objet 'args' attendu par les fonctions step_* de main.py."""
    defaults = dict(data="adult.data", model="xgboost", imbalance="class_weight", input=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def run(cmd):
    """Exécute une commande et lève une erreur si elle échoue."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec B603
    logger.info(res.stdout)
    if res.returncode != 0:
        logger.error(res.stderr)
        raise RuntimeError(f"Échec : {' '.join(cmd)}")


# ---------------- Tasks : code ----------------
@task(name="Installer les dépendances")
def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run([sys.executable, "-m", "black", "--line-length", "120", *FILES])


@task(name="Qualité du code")
def check_quality():
    run([sys.executable, "-m", "flake8", "--max-line-length=120", *FILES])


@task(name="Sécurité du code")
def check_security():
    run([sys.executable, "-m", "bandit", "-q", "-ll", *FILES])


@task(name="Tests unitaires")
def run_tests():
    run([sys.executable, "-m", "pytest", "-v"])


# ---------------- Tasks : données / modèle ----------------
@task(name="Préparation des données")
def prepare(args):
    m.step_prepare(args)


@task(name="Entraînement du modèle")
def train(args):
    # step_train entraîne ET sauvegarde le modèle
    m.step_train(args)


@task(name="Sauvegarder le modèle")
def save():
    # vérifie que le modèle a bien été sauvegardé par l'étape d'entraînement
    if not os.path.exists(m.MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {m.MODEL_PATH}")
    get_run_logger().info("Modèle sauvegardé : %s", m.MODEL_PATH)


@task(name="Charger le modèle")
def load():
    bundle = load_model(m.MODEL_PATH)
    get_run_logger().info("Modèle chargé : %s", bundle["nom"])


@task(name="Évaluer le modèle")
def evaluate(args):
    m.step_evaluate(args)


@task(name="Prédiction")
def predict(args):
    m.step_predict(args)


# ---------------- Flows ----------------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow(model: str = "xgboost", imbalance: str = "class_weight"):
    args = make_args(model=model, imbalance=imbalance)
    prepare(args)
    train(args)


@flow(name="evaluate")
def evaluate_flow():
    args = make_args()
    load()
    evaluate(args)


@flow(name="predict")
def predict_flow(input_file: str):
    predict(make_args(input=input_file))


@flow(name="all")
def all_flow(model: str = "xgboost", imbalance: str = "class_weight"):
    code_flow()  # code avant la préparation des données
    args = make_args(model=model, imbalance=imbalance)
    prepare(args)
    train(args)
    save()
    evaluate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", default="all", choices=["all", "train", "evaluate", "code", "install", "predict"])
    parser.add_argument("--input", help="CSV pour le flow predict")
    cli = parser.parse_args()

    if cli.flow == "predict":
        predict_flow(cli.input)
    else:
        {
            "all": all_flow,
            "train": train_flow,
            "evaluate": evaluate_flow,
            "code": code_flow,
            "install": install_flow,
        }[cli.flow]()
