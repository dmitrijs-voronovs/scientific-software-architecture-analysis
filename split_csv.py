import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

MAX_FILE_SIZE_MB = 90 * 1000 * 1000


def split_file(path: str | Path):
    file_path = Path(path)
    content = pd.read_csv(file_path)
    file_size = file_path.stat().st_size
    num_rows = content.shape[0]
    num_chunks = math.ceil(file_size / MAX_FILE_SIZE_MB)
    print(f"Splitting file {path} of size {file_size} into {num_chunks} chunks")
    chunk_size = math.ceil(num_rows / num_chunks)
    chunks = [content.iloc[i:i + chunk_size] for i in tqdm(range(0, num_rows, chunk_size), desc=f"Splitting chunks of {file_path}")]
    for i, chunk in enumerate(chunks):
        chunk.to_csv(file_path.with_stem(f"{file_path.stem}.{i}"), index=False)


def check_file_sizes():
    for file_path in Path("metadata/keywords/").glob("*.csv"):
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE_MB:
            print(f"File size: {file_path} | {size}")
            split_file(file_path)


if __name__ == "__main__":
    check_file_sizes()