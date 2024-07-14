from utils.paths import Paths


def get_golden_repos() -> list[str]:
    with open(Paths.GOLDEN_REPOS, "r", encoding="utf-8") as f:
        return f.read().splitlines()
