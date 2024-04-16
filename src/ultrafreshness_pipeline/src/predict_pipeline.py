import logging
import os

import nirvana_dl

from src.data import create_catboost_dataset, load_lstm_data, save_data
from src.entities import PredictPipelineParams, read_predict_pipeline_params
from src.model import catboost_inference, load_lstm, lstm_inference


def predict_pipeline():
    predict_pipeline_params = read_predict_pipeline_params(nirvana_dl.params())
    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logging.info(f"start predict pipeline with params {predict_pipeline_params}")
    model = load_lstm(predict_pipeline_params)
    logging.info(f"Model {type(model)} is loaded")
    lstm_data = load_lstm_data(
        predict_pipeline_params.lstm_data_path,
        predict_pipeline_params.cluster
    )
    logging.info(f"Data size {len(lstm_data)} is loaded")
    pred_probs = lstm_inference(model, lstm_data, predict_pipeline_params.device)
    logging.info("LSTM inference is done")

    catboost_data = create_catboost_dataset(
        predict_pipeline_params.catboost_data_path,
        predict_pipeline_params.cluster,
        pred_probs
    )
    ultrafreshness_probs = catboost_inference(
        predict_pipeline_params.catboost_path,
        catboost_data
    )
    logging.info(f"CatBoost inference is done")

    data_folder = nirvana_dl.output_data_path()
    status = save_data(data_folder, "result.json", ultrafreshness_probs)
    logging.info(f"Data is saved with status {status}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(nirvana_dl.logs_path(), "pipeline_log.py"),
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s"
    )
    predict_pipeline()
