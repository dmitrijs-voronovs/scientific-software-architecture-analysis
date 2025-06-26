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
    WEB = PROJECT_ROOT / "web"
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

    # KEYWORD DIRS
    KEYWORDS_MATCHING = KEYWORDS / FolderNames.KEYWORDS_MATCHING_DIR
    OPTIMIZED_KEYWORDS = KEYWORDS / FolderNames.OPTIMIZED_KEYWORD_DIR
    NOISE_FILTERING = KEYWORDS / FolderNames.NOISE_FILTERING_DIR
    QA_RELEVANCE_CHECK = KEYWORDS / FolderNames.QA_RELEVANCE_CHECK_DIR
    ARCH_RELEVANCE_CHECK = KEYWORDS / FolderNames.ARCH_RELEVANCE_CHECK_DIR
    TACTIC_EXTRACTION = KEYWORDS / FolderNames.TACTIC_EXTRACTION_DIR
