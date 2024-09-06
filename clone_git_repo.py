import os
import json
import subprocess
import tempfile
from pathlib import Path

from git import Repo
from pygit2 import Repository, GIT_SORT_TOPOLOGICAL


def clone_repo(author, repo_name):
    repo_url = f"https://github.com/{author}/{repo_name}.git"
    path = f"./.tmp/{author}/{repo_name}/master"
    os.makedirs(path, exist_ok=True)
    print(path)
    if not os.path.exists(path):
        Repo.clone_from(repo_url, path)
    return path


def run_gumtree_diff(author, repo_name, repo_path, tag1, tag2):
    # Create temporary directories for each tag
    path1 = f"./.tmp/{author}/{repo_name}/{tag1}"
    path2 = f"./.tmp/{author}/{repo_name}/{tag2}"
    print(f"{path1=}, {path2=}")

    # Clone the repository twice and checkout the respective tags
    if not os.path.exists(path1):
        Repo.clone_from(repo_path, path1).git.checkout(tag1)
    if not os.path.exists(path2):
        Repo.clone_from(repo_path, path2).git.checkout(tag2)

    full_path_to_file = Path(__file__).resolve().parent

    # Run GumTree on the entire project
    cmd = [
        "gumtree", "textdiff",
        "-f", "JSON",
        os.path.join(full_path_to_file, path1),
        os.path.join(full_path_to_file, path2),
        "-o", f"{repo_path}_{tag1}..{tag2}.diff.json"
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up temporary directories
    # subprocess.run(["rm", "-rf", path1, path2])

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
