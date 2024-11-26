from datetime import datetime

from utils.paths import Paths


def get_golden_repos() -> list[str]:
    with open(Paths.GOLDEN_REPOS, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def create_logger_path(prefix: str) -> str:
    return f".logs/{prefix}.{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"
