import html

import pandas as pd

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


class NoiseFilteringStageVerification(IStageVerification):
    stage_to_verify = NoiseFilteringStage

    source_columns = ['sentence']
    ai_output_columns = ['to_eliminate', 'reasoning']


def main():
    NoiseFilteringStageVerification(hostname=LLMHost.TECH_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
