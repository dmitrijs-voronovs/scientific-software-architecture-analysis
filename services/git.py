import os
from pathlib import Path
from typing import Optional

from git import Repo


def get_repo_base_path(author, repo_name, postfix="master"):
    repo_base_path = f"./.tmp/source/{author}/{repo_name}/{postfix}"
    return repo_base_path


def clone_repo(author, repo_name, postfix: Optional[str] = "master"):
    repo_url = f"https://github.com/{author}/{repo_name}.git"
    path = get_repo_base_path(author, repo_name, postfix)
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(path) or not os.listdir(path):
        Repo.clone_from(repo_url, path)
    return path


def clone_tag(author, repo_name, repo_path, tag1):
    path1 = f"./.tmp/source/{author}/{repo_name}/{tag1}"
    if not os.path.exists(path1) or not os.listdir(path1):
        Repo.clone_from(repo_path, path1).git.checkout(tag1)
    return path1


def checkout_tag(author, repo_name, tag) -> str:
    path1 = get_repo_base_path(author, repo_name)
    if not os.path.exists(path1) or not os.listdir(path1):
        clone_repo(author, repo_name)

    Repo(path1).git.checkout(tag)

    return path1


def get_abs_parent_dir():
    return Path(__file__).resolve().parent
