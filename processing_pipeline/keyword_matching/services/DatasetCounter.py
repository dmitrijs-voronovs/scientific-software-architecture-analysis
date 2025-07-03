from collections import defaultdict

import pandas as pd

from constants.abs_paths import AbsDirPath
from model.Credentials import Credentials
from processing_pipeline.keyword_matching.services.KeywordParser import MatchSource


class DatasetCounter:
    def __init__(self, run_id):
        self.datapoint_count_per_source = defaultdict(int)
        self.run_id = run_id
        self.filename = AbsDirPath.DATA / f"dataset_size/datapoints_per_source_{run_id}.csv"

    def add(self, repo: Credentials, source: MatchSource):
        self.datapoint_count_per_source[(repo.id, source.value)] += 1

    def reset(self):
        self.datapoint_count_per_source = defaultdict(int)
        self.filename.unlink(missing_ok=True)

    def save_datapoints_per_source_count(self):
        data_for_df = []
        for (repo_id, match_source), count in self.datapoint_count_per_source.items():
            data_for_df.append({"repo_id": repo_id, "match_source": match_source.value,
                                # Use .value if you want the string representation of the Enum
                                "count": count})
        df = pd.DataFrame(data_for_df)
        df_sorted = df.sort_values(by=["repo_id", "match_source"]).reset_index(drop=True)
        df_sorted.to_csv(self.filename, index=False)

    def restore_datapoints_per_source_count(self):
        if not self.filename.exists():
            print(f"Warning: File not found at {self.filename}. Returning empty defaultdict.")
            return

        try:
            df = pd.read_csv(self.filename)

            # Iterate over DataFrame rows and reconstruct the defaultdict
            for index, row in df.iterrows():
                repo_id = row["repo_id"]
                # Convert string back to MatchSource Enum member
                match_source_str = row["match_source"]
                try:
                    match_source = MatchSource(match_source_str)
                except ValueError:
                    print(
                        f"Warning: Unknown MatchSource '{match_source_str}' encountered for repo '{repo_id}'. Skipping.")
                    continue  # Skip this row if Enum conversion fails

                count = int(row["count"])  # Ensure count is an integer

                self.datapoint_count_per_source[(repo_id, match_source)] = count

            print(f"Data restored from: {self.filename}")

        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file {self.filename} is empty. Returning empty defaultdict.")
        except Exception as e:
            print(f"An error occurred while restoring data: {e}")
