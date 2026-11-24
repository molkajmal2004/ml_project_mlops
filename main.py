"""
main.py — Point d'entrée : exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --step prepare
    python main.py --step compare
    python main.py --step tune
    python main.py --step train --model xgboost --imbalance smote
    python main.py --step evaluate
    python main.py --step predict --input nouvelles_lignes.csv
    python main.py --step all      (ou simplement : python main.py --all)
"""

import argparse
import json
import os

import joblib
import pandas as pd

from model_pipeline import (
    COLS,
    IMBALANCE_METHODS,
    MODEL_NAMES,
    compare_models,
    evaluate_model,
    find_best_threshold,
    load_model,
    load_prepared_data,
    predict_new,
    prepare_data,
    save_model,
    train_model,
    tune_xgboost,
)

ARTIFACTS = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")
PARAMS_PATH = os.path.join(ARTIFACTS, "xgb_params.json")


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline ML — Adult (objectif C : has_capital_gain)")
    p.add_argument(
        "--step",
        choices=["prepare", "compare", "tune", "train", "evaluate", "predict", "all"],
        help="Étape à exécuter",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Exécute tout le pipeline d'un coup : prepare + train + evaluate (équivalent de --step all)",
    )
    p.add_argument("--data", default="adult.data", help="Fichier de données brutes")
    p.add_argument("--model", default="xgboost", choices=MODEL_NAMES, help="Modèle à entraîner")
    p.add_argument(
        "--imbalance",
        default="class_weight",
        choices=IMBALANCE_METHODS,
        help="Méthode de rééquilibrage des classes (train seulement)",
    )
    p.add_argument("--input", help="CSV de lignes brutes (sans en-tête, format adult.data) pour --step predict")
    return p.parse_args()


def step_prepare(args):
    X_tr, X_te, y_tr, y_te, _ = prepare_data(args.data, save_dir=ARTIFACTS)
    print(
        f"[prepare] Train : {X_tr.shape} | Test : {X_te.shape} | "
        f"% de 'oui' train : {y_tr.mean():.2%} | test : {y_te.mean():.2%}"
    )
    print(f"[prepare] Fichiers sauvegardés dans '{ARTIFACTS}/'")


def step_compare(args):
    X_tr, _, y_tr, _ = load_prepared_data(ARTIFACTS)
    print("[compare] Validation croisée (5 plis) sur le train — cela peut prendre quelques minutes...")
    print(compare_models(X_tr, y_tr, args.imbalance).round(3).to_string(index=False))


def step_tune(args):
    X_tr, _, y_tr, _ = load_prepared_data(ARTIFACTS)
    print("[tune] RandomizedSearchCV sur XGBoost (20 combinaisons)...")
    params, score = tune_xgboost(X_tr, y_tr)
    with open(PARAMS_PATH, "w") as f:
        json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}, f, indent=2)
    print(f"[tune] Meilleurs paramètres : {params} | PR-AUC (CV) = {score:.3f}")
    print(f"[tune] Sauvegardés dans {PARAMS_PATH} (utilisés automatiquement par --step train --model xgboost)")


def step_train(args):
    X_tr, _, y_tr, _ = load_prepared_data(ARTIFACTS)
    xgb_params = None
    if args.model == "xgboost" and os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            xgb_params = json.load(f)
        print(f"[train] Hyperparamètres réglés chargés depuis {PARAMS_PATH}")

    model = train_model(X_tr, y_tr, args.model, args.imbalance, xgb_params)
    seuil = find_best_threshold(model, X_tr, y_tr)
    save_model(model, MODEL_PATH, threshold=seuil, name=args.model, imbalance=args.imbalance)
    print(f"[train] Modèle '{args.model}' ({args.imbalance}) entraîné | seuil (F1 max, CV) = {seuil:.3f}")
    print(f"[train] Sauvegardé : {MODEL_PATH}")


def step_evaluate(args):
    _, X_te, _, y_te = load_prepared_data(ARTIFACTS)
    bundle = load_model(MODEL_PATH)
    print(f"[evaluate] Modèle : {bundle['nom']} ({bundle['desequilibre']})")
    for titre, seuil in [("Seuil 0,5", 0.5), ("Seuil optimisé", bundle["seuil"])]:
        m = evaluate_model(
            bundle["modele"],
            X_te,
            y_te,
            threshold=seuil,
            plots_dir=os.path.join(ARTIFACTS, "plots") if titre != "Seuil 0,5" else None,
        )
        print(f"\n--- {titre} ({m['seuil']}) ---")
        for k in ["accuracy", "precision", "rappel", "f1", "pr_auc", "roc_auc", "taux_de_base"]:
            print(f"{k:>14s} : {m[k]:.3f}")
        print("matrice de confusion :", m["matrice_confusion"])
    print(f"\n[evaluate] Graphiques : {ARTIFACTS}/plots/evaluation.png")


def step_predict(args):
    if not args.input:
        raise SystemExit("--step predict nécessite --input <fichier.csv>")
    df_raw = pd.read_csv(args.input, names=COLS, na_values="?", skipinitialspace=True)
    preprocess = joblib.load(os.path.join(ARTIFACTS, "preprocess.joblib"))
    print(predict_new(df_raw, preprocess, load_model(MODEL_PATH)).to_string())


# Fonctions de model_pipeline.py appelées par chaque étape (affichées pour la démonstration)
FONCTIONS = {
    "prepare": "prepare_data()  [load_data, clean_data, engineer_features, build_preprocessor]",
    "train": "train_model(), find_best_threshold(), save_model()",
    "evaluate": "load_model(), evaluate_model()",
}


def main():
    args = parse_args()
    steps = {
        "prepare": step_prepare,
        "compare": step_compare,
        "tune": step_tune,
        "train": step_train,
        "evaluate": step_evaluate,
        "predict": step_predict,
    }
    if not args.all and not args.step:
        raise SystemExit("Choisis une étape : --step <étape> ou --all (voir --help)")

    if args.all or args.step == "all":
        for i, s in enumerate(["prepare", "train", "evaluate"], 1):
            print(f"\n===== Étape {i}/3 : {s} -> {FONCTIONS[s]} =====")
            steps[s](args)
        print("\n===== Pipeline terminé : prepare -> train -> evaluate =====")
    else:
        steps[args.step](args)


if __name__ == "__main__":
    main()
