from pathlib import Path
from typing import List

import pandas as pd


def split_dataset_by_repo_and_source(out_dir: Path, df: pd.DataFrame, *, clean_before_saving=False, keep_index=False, drop_columns_before_save: List[str]=None):
    drop_columns_before_save = drop_columns_before_save or []
    all_columns_but_excluded = [col for col in df if col not in drop_columns_before_save]
    out_dir.mkdir(exist_ok=True)
    if clean_before_saving:
        for filepath in out_dir.iterdir():
            Path(filepath).unlink()

    for repo_id, source in df.drop_duplicates(["repo_id", "source"])[["repo_id", "source"]].values:
        output_file = out_dir / f"{repo_id.replace("/", ".")}.{source}.parquet"
        print(output_file)
        df[(df.source == source) & (df.repo_id == repo_id)][all_columns_but_excluded].to_parquet(output_file, engine='pyarrow',
                                                                    compression='snappy', index=keep_index)
        print(f"Saved {output_file}")


def load_all_files(in_dir: Path, *, name_contains: str = None):
    dfs = []
    for file_path in in_dir.glob("*.parquet"):
        if name_contains and not name_contains in file_path:
            continue
        try:
            file = pd.read_parquet(file_path, engine='pyarrow')
            # file['fname'] = file_path
            dfs.append(file)
            print(f"Loaded {file_path}")
        except:
            print(f"Error while loading {file_path}")

    df = pd.concat(dfs)
    return df
