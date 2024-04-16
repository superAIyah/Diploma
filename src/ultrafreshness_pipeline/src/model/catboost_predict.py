from typing import Any

import pandas as pd
from catboost import CatBoostClassifier


def load_catboost(model_path: str) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def catboost_inference(model_path: str, dataset: pd.core.frame.DataFrame) -> list[dict[Any, Any]]:
    model = load_catboost(model_path)
    keys = dataset.key.to_numpy()
    dataset = dataset.drop(["key", "query", "date"], axis=1)
    y_pred = model.predict_proba(dataset)[:, 1]
    result = []
    for key, y_prob in zip(keys, y_pred):
        result.append({
            "key": key,
            "ultrafreshness_prob": y_prob
        })
    return result
