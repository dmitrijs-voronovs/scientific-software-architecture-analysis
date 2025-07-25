from functools import cache

import pandas as pd
import yaml

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from cfg.tactics.tactic_list_simplified import TacticSimplifiedModelResponse
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage
from processing_pipeline.s3_tactic_extraction.TacticExtraction import TacticExtractionStage


def main():
    TacticExtractionStage(hostname=LLMHost.SERVER).execute(["issue", "issue_comment"], reverse=False)


if __name__ == "__main__":
    main()
