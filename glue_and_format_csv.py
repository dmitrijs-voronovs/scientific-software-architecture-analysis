import math
import os
from itertools import islice
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extract_quality_attribs_from_docs import MatchSource
from metadata.repo_info.repo_info import credential_list
from split_csv import check_file_sizes


def get_data(keywords_dir, cred, source: 'MatchSource'):
    dfs = []
    for file in keywords_dir.glob(f"{cred.dotted_ref}.{source.value}*.csv"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as error:
            print(f"Error reading {file}, {error=}")

    return pd.concat(dfs)


def transform_data(df):
    group_by_columns = ["quality_attribute", "sentence", "source", "author", "repo", "version"]
    transformations = df.groupby(group_by_columns).agg(
        total_similar=("id", "count"),
        target_keywords=("keyword", lambda x: sorted(x.unique())),
        target_matched_words=("matched_word", lambda x: sorted(x.unique())))
    res = df.groupby(group_by_columns).first().reset_index()
    res = res.merge(transformations, on=group_by_columns)
    # return df.sort_values(["total_similar", "sentence"], ascending=False)
    return res


def save_data(df, target_dir, cred, source: 'MatchSource'):
    os.makedirs(target_dir, exist_ok=True)
    df.to_csv(target_dir / f"{cred.dotted_ref}.{source.value}.csv", index=False)


def main():
    keywords_dir = Path("metadata/keywords")
    target_dir = keywords_dir / "optimized"

    for cred in tqdm(credential_list, desc="Processing keywords"):
        for source in MatchSource:
            tqdm.write(cred.dotted_ref)
            try:
                df = get_data(keywords_dir, cred, source)
                df = transform_data(df)
                save_data(df, target_dir, cred, source)
            except Exception as error:
                print(f"Error processing {cred.get_ref()}, {error=}")

    check_file_sizes(target_dir)


if __name__ == "__main__":
    main()