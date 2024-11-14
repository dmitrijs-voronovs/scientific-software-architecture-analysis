import math
from itertools import islice
from pathlib import Path

import pandas as pd
from tqdm import tqdm

MAX_FILE_SIZE_BYTES = 90 * 1000 * 1000


def split_file(path: str | Path, size_bytes=MAX_FILE_SIZE_BYTES):
    file_path = Path(path)
    content = pd.read_csv(file_path)
    file_size = file_path.stat().st_size
    num_rows = content.shape[0]
    num_chunks = math.ceil(file_size / size_bytes)
    print(f"Splitting file {path} of size {file_size} into {num_chunks} chunks")
    chunk_size_rows = math.ceil(num_rows / num_chunks)
    chunks = [content.iloc[ran] for ran in tqdm(grouper_ranges(num_rows, chunk_size_rows), desc=f"Splitting chunks of {file_path}")]
    print(chunks)
    for i, chunk in enumerate(chunks):
        chunk.to_csv(file_path.with_stem(f"{file_path.stem}.{i}"), index=False)


def check_file_sizes():
    for file_path in Path("metadata/keywords/").glob("*.csv"):
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            print(f"File size: {file_path} | {size}")
            split_file(file_path)


def grouper_ranges(total_size, chunk_size):
    return (range(i, min(i + chunk_size, total_size)) for i in range(0, total_size, chunk_size))



if __name__ == "__main__":
    # check_file_sizes()
    print(list(grouper_ranges(98, 10)))
    split_file("metadata/keywords/broadinstitute.cromwell.87.ISSUE_COMMENT.csv", 500_000)