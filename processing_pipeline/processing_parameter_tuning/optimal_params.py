import dataclasses
from typing import Dict

from cfg.ModelName import ModelName


@dataclasses.dataclass
class ProcessingParams:
    model_name: str
    n_threads: int
    batch_size: int


optimal_processing_parameters: Dict[str, ProcessingParams] = {
    p.model_name: p for p in [
        ProcessingParams(ModelName.DEEPSEEK_1_5B, 15, 10),
        ProcessingParams(ModelName.DEEPSEEK_7B, 10, 5),
        ProcessingParams(ModelName.DEEPSEEK_8B, 10, 50),
        ProcessingParams(ModelName.DEEPSEEK_14B, 10, 20),
    ]
}

# for chain of thoughts it is the same but half the batch and threads
optimal_processing_parameters_cot: Dict[str, ProcessingParams] = {
    p.model_name: p for p in [
        ProcessingParams(ModelName.DEEPSEEK_1_5B, 10, 20),
        ProcessingParams(ModelName.DEEPSEEK_7B, 6, 5),
        ProcessingParams(ModelName.DEEPSEEK_8B, 10, 20),
        ProcessingParams(ModelName.DEEPSEEK_14B, 6, 10),
    ]
}
