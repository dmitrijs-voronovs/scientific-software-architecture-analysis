from processing_pipeline.utilities.data_transformation import split_dataset_by_repo_and_source
from utilities.split_parquet import split_files_exceeding_max_limit
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Optional, Callable

import pandas as pd

from constants.abs_paths import AbsDirPath
from processing_pipeline.utilities.data_transformation import load_all_files

type ApplyFilters = Callable[[pd.DataFrame], pd.DataFrame]

COLUMNS_FOR_SPLITTING_DATA = ["repo_id", "source"]

@dataclass
class StageConfig:
    name: str
    depends_on_fields: List[str]
    resulting_fields: List[str]
    in_dir: Path
    out_dir: Path | None = None
    boolean_field_name: str | None = None
    next_stage: Optional['StageConfig'] = None
    _apply_filters: ApplyFilters = field(default_factory=lambda: (lambda df: df))

    def get_column_name(self, col_name: str):
        return f"{self.name}_{col_name}"

    def get_columns(self):
        return [self.get_column_name(column) for column in self.resulting_fields]

    def all_df_columns(self):
        return self.depends_on_fields + self.get_columns()

    def get_boolean_field(self):
        return self.get_column_name(self.boolean_field_name)

    @classmethod
    def load_main_df(cls):
        return load_all_files(AbsDirPath.O_KEYWORDS_MATCHING)

    def load_stage_df(self):
        return load_all_files(self.in_dir)

    def merge_stage_into_main(self, df_main: pd.DataFrame, df_stage: pd.DataFrame):
        return pd.merge(df_main, df_stage, "left", self.depends_on_fields)

    def get_passed_column(self):
        return f"{self.name}_passed"

    def apply_filters(self, df_stage: pd.DataFrame):
        df_stage_res = df_stage.copy()
        df_stage_res[self.get_passed_column()] = False
        df_stage_res.loc[self._apply_filters(df_stage_res).index, self.get_passed_column()] = True
        return df_stage_res

    def filter_passed(self, df_merged: pd.DataFrame):
        return df_merged[df_merged[self.get_passed_column()] == True]

    def apply_filters_till_current(self, keep_all_data = False):
        current_stage = first_stage
        df = self.load_main_df()
        while True:
            print(f"Applying filters for {current_stage.name}")
            df_stage = current_stage.load_stage_df()
            df_stage = current_stage.apply_filters(df_stage)
            if not keep_all_data:
                df_stage = current_stage.filter_passed(df_stage)
            df = current_stage.merge_stage_into_main(df, df_stage)

            if current_stage is self:
                break
            current_stage = current_stage.next_stage
        return df

    def optimize_df_for_next_stage(self, df: pd.DataFrame):
        next_stage_cfg = self.next_stage
        required_columns_for_the_next_stage_prompt = next_stage_cfg.depends_on_fields
        total_columns = required_columns_for_the_next_stage_prompt + COLUMNS_FOR_SPLITTING_DATA
        df = df.drop_duplicates(required_columns_for_the_next_stage_prompt)[total_columns]
        return df

    def save_data(self, df):
        split_dataset_by_repo_and_source(self.out_dir, df, clean_before_saving=True,
                                         drop_columns_before_save=COLUMNS_FOR_SPLITTING_DATA + [self.get_column_name("prompt")])
        split_files_exceeding_max_limit(self.out_dir)



S3TacticExtraction = StageConfig("s3", ["qa", "sentence"], ["tactic", "response"], AbsDirPath.S3_TACTIC_EXTRACTION,
                                 _apply_filters=lambda df: df[(~df['s3_tactic'].isna()) & (df['s3_tactic'] != "None")])

S2ArchRelevance = StageConfig("s2", ["sentence"], ["related_to_arch", "reasoning"], AbsDirPath.S2_ARCH_RELEVANCE_CHECK,
                              AbsDirPath.O_S2_ARCH_RELEVANCE_CHECK, "related_to_arch", S3TacticExtraction,
                              lambda df: df[df['s2_related_to_arch'] == True])

S1QARelevance = StageConfig("s1", ["qa", "sentence"], ["true_positive", "reasoning"], AbsDirPath.S1_QA_RELEVANCE_CHECK,
                            AbsDirPath.O_S1_QA_RELEVANCE_CHECK, "true_positive", S2ArchRelevance,
                            lambda df: df[df['s1_true_positive'] == True])

S0NoiseFiltering = StageConfig("s0", ["sentence"], ["to_eliminate", "reasoning"], AbsDirPath.S0_NOISE_FILTERING,
                               AbsDirPath.O_S0_NOISE_FILTERING, "to_eliminate", S1QARelevance,
                               lambda df: df[df['s0_to_eliminate'] == False])


def ps_apply_filters(df):
    df_res = df.copy()
    df_res = df_res.drop_duplicates(["sentence"])["sentence"]
    df_res['nwords'] = df.sentence.str.count(" ") + 1
    mask = df_res.nwords > 5
    df_res = df_res.drop(columns=["nwords"])
    return df_res[mask]


PrefilterStage = StageConfig("prefilter", ["sentence"], [], AbsDirPath.O_KEYWORDS_MATCHING, AbsDirPath.O2_KEYWORDS_MATCHING,
                             "to_eliminate", S0NoiseFiltering, ps_apply_filters)

first_stage = PrefilterStage
