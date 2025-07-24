from abc import ABC, abstractproperty
from typing import Literal

from pydantic import BaseModel

from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaFormatValidityResponse(BaseModel):
    correctness: Literal["correct", "partially correct", "incorrect"]
    reasoning: str


class IStageVerification(IBaseStage, ABC):
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    in_dir = AbsDirPath.SAMPLES
    out_dir = AbsDirPath.SAMPLES_VERIFIED
    cache_dir = AbsDirPath.CACHE / "samples"
    DFHandler = CSVDFHandler()
    data_model = OllamaFormatValidityResponse

    stage_to_verify = type[IBaseStage]

    @property
    def stage_name(self) -> str:
        return self.stage_to_verify.stage_name + '_v'

    def execute_verification(self):
        self.execute(self.stage_to_verify.stage_name)
