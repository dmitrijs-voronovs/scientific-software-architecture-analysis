import pandas as pd

from processing_pipeline.data_optimization.stage_cfg import last_stage, StageConfig


class FinalStageCleanup:
    @staticmethod
    def fix_passed_fields(df: pd.DataFrame):
        """
        """
        for cur in StageConfig.all_stages():
            df[cur.column_stage_passed] = df[cur.column_stage_passed].fillna(False).astype(bool)

    @staticmethod
    def fix_data_quality(df: pd.DataFrame):
        """
        Removes all data from the stage if it hasn't passed the funnel.
        """
        stage_it = StageConfig.all_stages()
        prev = next(stage_it)
        for cur in stage_it:
            df.loc[df[prev.column_stage_passed != True], cur.all_columns] = pd.NA

