import dataclasses
from typing import Any

from marshmallow_dataclass import class_schema

from .lstm_params import LstmParams


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PredictPipelineParams:
    lstm_path: str
    device: str
    lstm_data_path: str
    lstm_params: LstmParams
    catboost_path: str
    catboost_data_path: str
    cluster: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(params_d: dict[str, Any]) -> PredictPipelineParams:
    schema = PredictPipelineParamsSchema()
    return schema.load(params_d)
