import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class CredentialsDTO():
    author: str
    repo: str
    version: str
    wiki: Optional[str] = None

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    @property
    def repo_path(self) -> str:
        return f"{self.author}/{self.repo}"

    @property
    def repo_name(self) -> str:
        return f"{self.author}.{self.repo}"

    @property
    def wiki_dir(self) -> str:
        return re.split(r"https?://", self.wiki)[-1]

    def has_wiki(self) -> bool:
        return self.wiki is not None

    def _get_ref(self, delimiter="/") -> str:
        return delimiter.join([self.author, self.repo, self.version])

    @property
    def dotted_ref(self) -> str:
        return self._get_ref(".")

    @property
    def id(self) -> str:
        return self.id
