from typing import Any

import numpy as np
import torch

from src.data import create_lstm_dataset, create_lstm_loader
from src.entities import PredictPipelineParams

from .lstm import LSTMClassifier


def load_lstm(predict_pipeline_params: PredictPipelineParams) -> LSTMClassifier:
    model = LSTMClassifier(
        predict_pipeline_params.lstm_params.input_dim,
        predict_pipeline_params.lstm_params.hidden_dim,
        predict_pipeline_params.lstm_params.layer_dim,
        predict_pipeline_params.lstm_params.output_dim,
    )
    model.load_state_dict(torch.load(
        predict_pipeline_params.lstm_path,
        map_location=torch.device(predict_pipeline_params.device)
    ))
    model = model.to(predict_pipeline_params.device)
    model.eval()
    return model


def to_cpu(tens: torch.Tensor) -> np.array:
    return tens.detach().cpu().numpy()


def lstm_inference(
        model: PredictPipelineParams,
        dataset: tuple[dict[str, list[int]]],
        device: str
) -> dict[Any]:
    probs_total = []
    keys_total = []

    eval_ds = create_lstm_dataset(dataset)
    eval_dl = create_lstm_loader(eval_ds)
    with torch.no_grad():
        for x_val, keys in eval_dl:
            x_val.to(device)
            out = model(x_val, device)
            probs_total.extend(to_cpu(torch.sigmoid(out).squeeze(1)))
            keys_total.extend(keys)

    result = []
    for key, prob in zip(keys_total, probs_total):
        result.append({
            "key": key,
            "lstm_prob": float(prob)
        })
    return result
