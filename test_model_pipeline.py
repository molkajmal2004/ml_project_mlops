import os

import main


def test_paths_defined():
    assert main.MODEL_PATH.endswith("model.joblib")


def test_data_file_exists():
    assert os.path.exists("adult.data")


def test_imbalance_methods_available():
    from model_pipeline import IMBALANCE_METHODS, MODEL_NAMES
    assert len(IMBALANCE_METHODS) > 0
    assert "xgboost" in MODEL_NAMES