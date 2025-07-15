import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.BaseStage import BaseStage
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    NoiseFilteringStage(hostname=LLMHost.SERVER, disable_cache=True).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
    