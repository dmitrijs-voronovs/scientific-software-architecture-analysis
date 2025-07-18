from abc import ABC

from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.model.IBaseStage import IBaseStage


class IStageVerification(IBaseStage, ABC):
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    in_dir = AbsDirPath.SAMPLES
    out_dir = AbsDirPath.SAMPLES_VERIFIED
    cache_dir = AbsDirPath.CACHE / "samples"
    DFHandler = CSVDFHandler()
