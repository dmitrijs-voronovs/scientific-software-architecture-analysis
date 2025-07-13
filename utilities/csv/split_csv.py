import math
import os
from itertools import islice
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def split_file(file_path: Path, size_bytes=MAX_FILE_SIZE_BYTES / 2):
    content = pd.read_parquet(file_path)
    file_size = file_path.stat().st_size
    num_rows = content.shape[0]
    num_chunks = math.ceil(file_size / size_bytes)
    print(f"Splitting file {file_path} of size {file_size} into {num_chunks} chunks")
    chunk_size_rows = math.ceil(num_rows / num_chunks)
    chunks = [content.iloc[ran] for ran in grouper_ranges(num_rows, chunk_size_rows)]
    n_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.to_parquet(file_path.with_stem(f"{file_path.stem}.{i:0{len(str(n_chunks))}}"), engine='pyarrow', compression='snappy', index=False)


def split_files_exceeding_max_limit(dir, size_limit=MAX_FILE_SIZE_BYTES):
    for file_path in Path(dir).glob("*[A-Z].parquet"):
        size = file_path.stat().st_size
        if size > size_limit:
            print(f"File size: {file_path} | {size}")
            split_file(file_path)
            file_path.rename(file_path.with_stem(f"_{file_path.stem}"))


def grouper_ranges(total_size, chunk_size):
    return (range(i, min(i + chunk_size, total_size)) for i in range(0, total_size, chunk_size))



if __name__ == "__main__":
    # split_files_exceeding_max_limit(AbsDirPath.KEYWORDS_MATCHING)
    split_files_exceeding_max_limit(AbsDirPath.OPTIMIZED_KEYWORDS)
    # split_files_exceeding_max_limit(AbsDirPath.S0_NOISE_FILTERING)
    # split_files_exceeding_max_limit(AbsDirPath.S1_QA_RELEVANCE_CHECK)
    # split_files_exceeding_max_limit(AbsDirPath.S2_ARCH_RELEVANCE_CHECK)
    # split_files_exceeding_max_limit(AbsDirPath.S3_TACTIC_EXTRACTION)

    # print(list(grouper_ranges(98, 10)))
    # split_file("metadata/keywords/broadinstitute.cromwell.87.ISSUE_COMMENT.csv", 500_000)