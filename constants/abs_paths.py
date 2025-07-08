import os
from pathlib import Path

from constants.foldernames import FolderNames

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class AbsDirPath:
    """
    Defines absolute file system paths for key directories within the project.
    All paths are constructed relative to the dynamically determined PROJECT_ROOT.
    """
    # DIRS

    DATA = PROJECT_ROOT / "data"
    RESULTS = PROJECT_ROOT / "results"
    DOCS = PROJECT_ROOT / "docs"
    CONFIG = PROJECT_ROOT / "cfg"
    CONSTANTS = PROJECT_ROOT / "constants"
    RESOURCES = PROJECT_ROOT / "resources"
    STAGES = PROJECT_ROOT / "stages"

    TEMP = PROJECT_ROOT / ".tmp"
    CACHE = PROJECT_ROOT / ".cache"
    LOGS = PROJECT_ROOT / ".logs"

    # SUB DIRS
    QUERIES = RESOURCES / "queries"
    KEYWORDS = DATA / "keywords"
    SAMPLES = DATA / "samples"
    ANALYSIS = STAGES / "analysis"

    WIKIS = TEMP / "docs"
    SOURCE_CODE = TEMP / "source"

    # ANALYSIS DIRS
    REPO_TOPICS = ANALYSIS / "repo_topics"
    REPOS = ANALYSIS / "repos"
    KEYWORD_ANALYSIS = ANALYSIS / "keywords"

    # KEYWORD DIRS
    KEYWORDS_MATCHING = KEYWORDS / FolderNames.KEYWORDS_MATCHING_DIR
    OPTIMIZED_KEYWORDS = KEYWORDS / FolderNames.OPTIMIZED_KEYWORD_DIR
    S0_NOISE_FILTERING = KEYWORDS / FolderNames.NOISE_FILTERING_DIR
    S1_QA_RELEVANCE_CHECK = KEYWORDS / FolderNames.QA_RELEVANCE_CHECK_DIR
    S2_ARCH_RELEVANCE_CHECK = KEYWORDS / FolderNames.ARCH_RELEVANCE_CHECK_DIR
    S3_TACTIC_EXTRACTION = KEYWORDS / FolderNames.TACTIC_EXTRACTION_DIR

    OPTIMIZED_KEYWORDS_OLD = KEYWORDS / FolderNames.OPTIMIZED_KEYWORD_OLD_DIR
    PARAMETER_TUNING_DIR = KEYWORDS / FolderNames.PARAMETER_TUNING_DIR
    PARAMETER_TUNING_RES_DIR = KEYWORDS / FolderNames.PARAMETER_TUNING_RES_DIR

    # RESULTS DIRS
    RES_KEYWORDS_MATCHING = RESULTS / FolderNames.RES_KEYWORDS_MATCHING_DIR
    RES_S0_NOISE_FILTERING = RESULTS / FolderNames.RES_NOISE_FILTERING_DIR
    RES_S1_QA_RELEVANCE_CHECK = RESULTS / FolderNames.RES_QA_RELEVANCE_CHECK_DIR
    RES_S2_ARCH_RELEVANCE_CHECK = RESULTS / FolderNames.RES_ARCH_RELEVANCE_CHECK_DIR
    RES_S3_TACTIC_EXTRACTION = RESULTS / FolderNames.RES_TACTIC_EXTRACTION_DIR
