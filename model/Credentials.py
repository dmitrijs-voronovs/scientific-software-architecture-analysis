import functools
import re
from typing import Dict, Optional


class Credentials(Dict):
    author: str
    repo: str
    version: str
    wiki: Optional[str]

    @property
    def repo_path(self) -> str:
        return f"{self['author']}/{self['repo']}"

    @property
    def repo_name(self) -> str:
        return f"{self['author']}.{self['repo']}"

    @property
    def wiki_dir(self) -> str:
        return re.split(r"https?://", self["wiki"])[-1]

    def has_wiki(self) -> bool:
        return self["wiki"] is not None

    def get_ref(self, delimiter="/") -> str:
        return f"{self['author']}{delimiter}{self['repo']}{delimiter}{self['version']}"

    @property
    def dotted_ref(self) -> str:
        return self.get_ref(".")
