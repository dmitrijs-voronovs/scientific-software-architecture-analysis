import math
from pathlib import Path

from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.model.IDFHandler import IDfHandler
from processing_pipeline.model.ParquetDFHandler import ParquetDFHandler


def split_file(path: Path, n_parts: int, handler: IDfHandler):
    df = handler.read_df(path)
    for i in range(n_parts):
        handler.write_df(df.iloc[i::n_parts], path.with_stem(f"{path.stem}.pt_{i:0{len(str(n_parts))}}"))
    path.rename(path.with_stem(f"_{path.stem}"))

def split_file_in_batches(path: Path, handler: IDfHandler, batch_size: int = 1000):
    df = handler.read_df(path)
    n_parts = math.ceil(df.shape[0] / batch_size)
    split_file(path, n_parts, handler)

def split_file_in_seq_batches(path: Path, handler: IDfHandler, batch_size: int = 3000):
    """
    First batch keeps the name of the original file, thus saves related processing caches. Subsequent batches are
    named with a suffix and can be processed separately in parallel.

    Useful for
    - files that already started processing and are not yet completed.
    - files from the same source that are being processed by a previous stage. First batches of that same file can be
    processed won't be intact by the current stage.
    """
    df = handler.read_df(path)
    path.rename(path.with_stem(f"_{path.stem}"))

    n_parts = math.ceil(df.shape[0] / batch_size)
    print(f"{path}: Splitting into {n_parts} parts")
    for i in range(n_parts):
        start = i * batch_size
        part_name = path if i == 0 else path.with_stem(f"{path.stem}.pt_{i+1:0{len(str(n_parts))}}")
        handler.write_df(df.iloc[start:start + batch_size], part_name)


def main():
    # split_file(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\samples\verified\s0.csv"), 4, CSVDFHandler())
    # split_file_in_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s1_qa_relevance_check_o\root-project.root.v6-32-06.code_comment.0.parquet"), ParquetDFHandler())
    # split_file_in_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s1_qa_relevance_check_o\root-project.root.v6-32-06.code_comment.1.parquet"), ParquetDFHandler())
    # split_file_in_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s1_qa_relevance_check_o\root-project.root.v6-32-06.code_comment.2.parquet"), ParquetDFHandler())
    # split_file_in_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s1_qa_relevance_check_o\root-project.root.v6-32-06.code_comment.3.parquet"), ParquetDFHandler())
    # split_file_in_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s1_qa_relevance_check_o\root-project.root.v6-32-06.code_comment.4.parquet"), ParquetDFHandler())
    split_file_in_seq_batches(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\keywords_2\s2_arch_relevance_check_o\allenai.scispacy.v0.5.5.issue.parquet"), ParquetDFHandler(), 100)

if __name__ == "__main__":
    main()