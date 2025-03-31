import os
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from constants.foldernames import FolderNames
from extract_quality_attribs_from_docs import MatchSource
from metadata.repo_info.repo_info import credential_list
from model.Credentials import Credentials
from split_csv import split_files_exceeding_max_limit



def get_data(keywords_dir):
    dfs = []
    for file in keywords_dir.glob(f"*.csv"):
        try:
            print(f"reading file {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as error:
            print(f"Error reading {file}, {error=}")

    return pd.concat(dfs)

def save_data(df, target_filename: Path):
    os.makedirs(target_filename.parent, exist_ok=True)
    df.to_csv(target_filename, index=False)

def main():
    keywords_dir = Path(f"metadata/keywords/{FolderNames.FORMAT_VALIDATION_DIR}")
    target_filename = Path(f"metadata/results/format_validation_sample.csv")

    try:
        df = get_data(keywords_dir)
        save_data(df, target_filename)
    except Exception as error:
        print(f"Error processing {error=}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
