import os
import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from quality_attributes import quality_attributes_sample, QualityAttributesMap, quality_attributes
from services.ast_extractor import ext_to_lang, get_comments

AttributeDictType = Dict[str, List[str]]


class TextMatch(Dict):
    keyword: str
    matched_word: str
    sentence: str
    quality_attribute: str


class MatchSource(Enum):
    RELEASES = "RELEASES"
    WIKI = "WIKI"
    DOCS = "DOCS"
    ISSUE = "ISSUE"
    ISSUE_COMMENT = "ISSUE_COMMENT"
    CODE_COMMENT = "CODE_COMMENT"


class Credentials(Dict):
    author: str
    repo: str
    version: str

    @property
    def repo_path(self) -> str:
        return f"{self['author']}/{self['repo']}"

    @property
    def repo_name(self) -> str:
        return f"{self['author']}.{self['repo']}"

    def get_ref(self, delimiter="/") -> str:
        return f"{self['author']}{delimiter}{self['repo']}{delimiter}{self['version']}"


class FullMatch(TextMatch, Credentials):
    filename: str
    source: MatchSource
    url: str


def text_keyword_iterator(text: str, attributes: AttributeDictType) -> Generator[TextMatch, None, None]:
    sentences = re.split(r'(\r?\n|\.)', text)
    for quality_attr, keywords in attributes.items():
        for sentence in sentences:
            pattern = get_keyword_matching_pattern(keywords)
            match = re.search(pattern, sentence)
            if match:
                full_match, keyword = match.group(), match.group(1)
                yield TextMatch(quality_attribute=quality_attr, keyword=keyword, matched_word=full_match,
                                sentence=sentence.strip())


def get_keyword_matching_pattern(keywords):
    return re.compile(rf'\b({"|".join(keywords)})[\w-]*')


def strip_html_tags(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def generate_text_fragment_link(base_url: str, text: str, page: str = "") -> str:
    full_url = f"{base_url}/{page}" if page else base_url
    encoded_text = urllib.parse.quote(text)
    return f"{full_url}#:~:text={encoded_text}"


def parse_wiki(wiki_path: str, creds: Credentials, quality_attributes_map: QualityAttributesMap, wiki_url: str) -> List[FullMatch]:
    matches = []
    files = Path(wiki_path).glob("**/*.html")
    for file in tqdm(files, desc="Parsing wiki"):
        tqdm.write(str(file))
        abs_path = file
        rel_path = os.path.normpath(os.path.relpath(abs_path, start=wiki_path)).replace("\\", "/")
        documentation_raw = open(abs_path, "r", encoding="utf-8").read()
        text_content = strip_html_tags(documentation_raw)
        matches.extend(
            [FullMatch(**match, source=MatchSource.WIKI, filename=rel_path, **creds,
                       url=generate_text_fragment_link(wiki_url, match.get("sentence"), rel_path)) for
             match in
             text_keyword_iterator(text_content, quality_attributes_map)])

    return matches


def parse_docs(docs_path: str, creds: Credentials, quality_attributes_map: QualityAttributesMap) -> List[FullMatch]:
    repo_url = get_github_repo_url(creds)
    matches = []
    docs_extensions = [".md", ".rst", ".txt", ".adoc", ".html"]
    for ext in docs_extensions:
        files = Path(docs_path).glob(f"**/*{ext}")
        for file in tqdm(files, desc="Parsing docs"):
            tqdm.write(str(file))
            abs_path = file
            rel_path = os.path.normpath(os.path.relpath(abs_path, start=docs_path)).replace("\\", "/")
            documentation_raw = open(abs_path, "r", encoding="utf-8").read()
            text_content = strip_html_tags(documentation_raw) if ext in ".html" else documentation_raw
            matches.extend(
                [FullMatch(**match, source=MatchSource.DOCS, filename=rel_path, **creds,
                           url=generate_text_fragment_link(repo_url, match.get("sentence"), rel_path)) for
                 match in
                 text_keyword_iterator(text_content, quality_attributes_map)])

    return matches


BASE_GITHUB_URL = "https://github.com"


def get_github_repo_url(creds: Credentials) -> str:
    return f"{BASE_GITHUB_URL}/{creds['author']}/{creds['repo']}/tree/{creds['version']}"


def parse_comments(source_code_path: str, creds: Credentials, quality_attributes_map: QualityAttributesMap) -> List[FullMatch]:
    repo_url = get_github_repo_url(creds)
    matches = []
    for root, dirs, files in tqdm(os.walk(source_code_path), desc="Parsing code comments"):
        tqdm.write(str(root))
        for file in files:
            supported_languages = ext_to_lang.keys()
            if any(file.endswith(ext) for ext in supported_languages):
                abs_path = os.path.join(root, file)
                rel_path = os.path.normpath(os.path.relpath(abs_path, source_code_path)).replace("\\", "/")
                text_content = "\n".join(get_comments(abs_path))
                matches.extend(
                    [FullMatch(**match, source=MatchSource.CODE_COMMENT, filename=rel_path, **creds,
                               url=generate_text_fragment_link(repo_url, match.get("sentence"), rel_path)) for
                     match in
                     text_keyword_iterator(text_content, quality_attributes_map)])

    return matches


def save_to_file(records: List[FullMatch], source: MatchSource, creds: Credentials):
    dir = Path("metadata") / "keywords"
    os.makedirs(dir, exist_ok=True)
    filename = f'{creds.get_ref(".")}.{source.value}.csv'
    pd.DataFrame(records).to_csv(dir / filename, index=False)


if __name__ == "__main__":
    creds = Credentials(author="scverse", repo="scanpy", version="1.10.2")
    wiki_url = "scanpy.readthedocs.io/en"
    protocol = "https://"
    docs_path = Path(".tmp/docs")
    source_code_path = Path(".tmp/source")
    matches_wiki = parse_wiki(str(docs_path / f'{creds.repo_path}/{wiki_url}'), creds, quality_attributes,
                              f'{protocol}{wiki_url}')
    save_to_file(matches_wiki, MatchSource.WIKI, creds)

    matches_code_comments = parse_comments(str(source_code_path / creds.get_ref()), creds, quality_attributes)
    save_to_file(matches_code_comments, MatchSource.CODE_COMMENT, creds)

    matches_docs = parse_docs(str(source_code_path / creds.get_ref()), creds, quality_attributes)
    save_to_file(matches_docs, MatchSource.DOCS, creds)
