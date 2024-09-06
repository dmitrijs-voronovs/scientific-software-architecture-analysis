import os
from pathlib import Path

from git import Repo


def clone_repo(author, repo_name):
    repo_url = f"https://github.com/{author}/{repo_name}.git"
    path = f"./.tmp/{author}/{repo_name}/master"
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(path):
        Repo.clone_from(repo_url, path)
    return path


def clone_tag(author, repo_name, repo_path, tag1):
    path1 = f"./.tmp/{author}/{repo_name}/{tag1}"
    if not os.path.exists(path1):
        Repo.clone_from(repo_path, path1).git.checkout(tag1)
    return path1


def get_abs_parent_dir():
    return Path(__file__).resolve().parent
