import dataclasses
from pathlib import Path
from typing import List, Optional

from constants.abs_paths import AbsDirPath


@dataclasses.dataclass
class StageConfig:
    name: str
    depends_on_fields: List[str]
    resulting_fields: List[str]
    data_dir: Path
    boolean_field_name: str | None = None
    next_stage: Optional['StageConfig'] = None

    def get_columns(self):
        return [f"{self.name}_{column}" for column in self.resulting_fields]

    def all_df_columns(self):
        return self.depends_on_fields + self.get_columns()

S3TacticExtraction = StageConfig("s3", ["sentence"], ["tactic", "response"], AbsDirPath.S3_TACTIC_EXTRACTION)
S2ArchRelevance = StageConfig("s2", ["sentence"], ["related_to_arch", "reasoning"], AbsDirPath.S2_ARCH_RELEVANCE_CHECK, "related_to_arch", next_stage=S3TacticExtraction)
S1QARelevance = StageConfig("s1", ["qa", "sentence"], ["true_positive", "reasoning"], AbsDirPath.S1_QA_RELEVANCE_CHECK, "true_positive", next_stage=S2ArchRelevance)
S0NoiseFiltering = StageConfig("s0", ["sentence"],  ["to_eliminate", "reasoning"], AbsDirPath.S0_NOISE_FILTERING, "to_eliminate", next_stage=S1QARelevance)





