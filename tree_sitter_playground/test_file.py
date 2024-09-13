from typing import List
from services.git import clone_repo, clone_tag, get_abs_parent_dir

import os
import json
import pathlib as pppp
import subprocess
from re import sub as sssub


# WORDS = "1,2,31,4124,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20".split(",")
any_book = 1
MY_FAVORITE_BOOK = "Garfield"
WORDS = "1,2,31,4124,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20".split(",")

type Book = List[int]


class BaseReader:
    """
    Base class for reading books
    """
    def __init__(self, book: Book):
        self.book = book

    def read(self, page: int) -> int:
        return self.book[page]

    @classmethod
    def create_class(cls):
        return BaseReader([1, 23])

    @staticmethod
    def read_all_words(cls):
        return ",".join(str(word) for word in cls.book)

    def read_all(self):
        return self.book


class WordReader(BaseReader):
    instances = 0
    new_books: List[Book]

    def read(self, page: int):
        return self.book[page]

    def read_all(self):
        return self.book

    def read_all_words(self):
        return ",".join(str(word) for word in self.book)


class Writer():
    def write(self, word: str):
        print(word)


class Manager(BaseReader, Writer):
    def test(self):
        pass


def run_gumtree_diff(author: str, repo_name, repo_path, tag1, tag2) -> dict:
    # Clone the repository twice and checkout the respective tags
    path1 = clone_tag(author, repo_name, repo_path, tag1)
    path2 = clone_tag(author, repo_name, repo_path, tag2)

    root_dir = get_abs_parent_dir()

    # Run GumTree on the entire project
    cmd = [
        "gumtree", "textdiff",
        "-f", "JSON",
        os.path.join(root_dir, path1),
        os.path.join(root_dir, path2),
        "-o", f"{repo_path}_{tag1}..{tag2}.diff.json"
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    return json.loads(result.stdout)


def generate_metadata(author, repo_name, tag1, tag2):
    os.makedirs('./.tmp', exist_ok=True)
    repo_path = clone_repo(author, repo_name)

    metadata = {
        "repo": f"{author}/{repo_name}",
        "tag1": tag1,
        "tag2": tag2,
        "changes": run_gumtree_diff(author, repo_name, repo_path, tag1, tag2)
    }

    return metadata


def main():
    def sub_fun():
        pass

    # author = input("Enter GitHub repository author: ")
    # repo_name = input("Enter GitHub repository name: ")
    # tag1 = input("Enter first tag: ")
    # tag2 = input("Enter second tag: ")

    author = "scverse"
    repo_name = "scanpy"
    tag1 = "1.10.1"
    tag2 = "1.10.2"

    metadata = generate_metadata(author, repo_name, tag1, tag2)

    with open("diff_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata has been saved to diff_metadata.json")


if __name__ == "__main__":
    main()
