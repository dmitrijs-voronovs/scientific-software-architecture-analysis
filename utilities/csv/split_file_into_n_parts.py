from pathlib import Path

from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.model.IDFHandler import IDfHandler


def split_file(path: Path, n_parts: int, handler: IDfHandler):
    df = handler.read_df(path)
    for i in range(n_parts):
        handler.write_df(df.iloc[i::n_parts], path.with_stem(f"{path.stem}.part_{i:0{len(str(n_parts))}}"))


def main():
    split_file(Path(r"C:\Users\Dmitrijs\Documents\myDocs\masters\courses\thesis\code\data\samples\verified\s0.csv"), 4, CSVDFHandler())

if __name__ == "__main__":
    main()