import os
import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator, Optional

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from model.Credentials import Credentials
from quality_attributes import quality_attributes
from services.ast_extractor import ext_to_lang, get_comments

AttributeDictType = Dict[str, List[str]]


class TextMatch(Dict):
    keyword: str
    matched_word: str
    sentence: str
    quality_attribute: str
    text: Optional[str]


class MatchSource(Enum):
    RELEASES = "RELEASES"
    WIKI = "WIKI"
    DOCS = "DOCS"
    ISSUE = "ISSUE"
    ISSUE_COMMENT = "ISSUE_COMMENT"
    CODE_COMMENT = "CODE_COMMENT"


class FullMatch(TextMatch, Credentials):
    filename: str
    source: MatchSource
    url: str


BASE_GITHUB_URL = "https://github.com"


class KeywordParser:
    def __init__(self, attributes: AttributeDictType, creds: Credentials, *, append_full_text: bool = False):
        self.attributes = attributes
        self.creds = creds
        self.append_full_text = append_full_text

    def _text_keyword_iterator(self, text: str) -> Generator[TextMatch, None, None]:
        sentences = re.split(r'(\r?\n|\.)', text)
        for quality_attr, keywords in self.attributes.items():
            for sentence in sentences:
                pattern = self.get_keyword_matching_pattern(keywords)
                match = re.search(pattern, sentence)
                if match:
                    full_match, keyword = match.group(), match.group(1)
                    if self.append_full_text:
                        yield TextMatch(quality_attribute=quality_attr, keyword=keyword, matched_word=full_match,
                                        sentence=sentence.strip(), text=text if self.append_full_text else None)
                    else:
                        yield TextMatch(quality_attribute=quality_attr, keyword=keyword, matched_word=full_match,
                                        sentence=sentence.strip())

    @staticmethod
    def get_keyword_matching_pattern(keywords):
        return re.compile(rf'\b({"|".join(keywords)})[A-Za-z-]*\b')

    @staticmethod
    def _strip_html_tags(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    @staticmethod
    def _generate_text_fragment_link(base_url: str, text: str, page: str = "") -> str:
        full_url = f"{base_url}/{page}" if page else base_url
        encoded_text = urllib.parse.quote(text)
        return f"{full_url}#:~:text={encoded_text}"

    def parse_wiki(self, wiki_path: str) -> List[FullMatch]:
        matches = []
        files = Path(wiki_path).glob("**/*.html")
        for file in tqdm(files, desc="Parsing wiki"):
            tqdm.write(str(file))
            abs_path = file
            rel_path = os.path.normpath(os.path.relpath(abs_path, start=wiki_path)).replace("\\", "/")
            documentation_raw = open(abs_path, "r", encoding="utf-8").read()
            text_content = self._strip_html_tags(documentation_raw)
            matches.extend(
                [FullMatch(**match, source=MatchSource.WIKI, filename=rel_path, **self.creds,
                           url=self._generate_text_fragment_link(self.creds['wiki'], match.get("sentence"), rel_path)) for
                 match in
                 self._text_keyword_iterator(text_content)])

        return matches

    def parse_docs(self, docs_path: str) -> List[FullMatch]:
        repo_url = self._get_github_repo_url()
        matches = []
        docs_extensions = [".md", ".rst", ".txt", ".adoc", ".html"]
        for ext in docs_extensions:
            files = Path(docs_path).glob(f"**/*{ext}")
            for file in tqdm(files, desc="Parsing docs"):
                tqdm.write(str(file))
                abs_path = file
                rel_path = os.path.normpath(os.path.relpath(abs_path, start=docs_path)).replace("\\", "/")
                documentation_raw = open(abs_path, "r", encoding="utf-8").read()
                text_content = self._strip_html_tags(documentation_raw) if ext in ".html" else documentation_raw
                matches.extend(
                    [FullMatch(**match, source=MatchSource.DOCS, filename=rel_path, **self.creds,
                               url=self._generate_text_fragment_link(repo_url, match.get("sentence"), rel_path)) for
                     match in
                     self._text_keyword_iterator(text_content)])

        return matches

    def _get_github_repo_url(self) -> str:
        return f"{BASE_GITHUB_URL}/{self.creds['author']}/{self.creds['repo']}/tree/{self.creds['version']}"

    def parse_comments(self, source_code_path: str) -> List[FullMatch]:
        repo_url = self._get_github_repo_url()
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
                        [FullMatch(**match, source=MatchSource.CODE_COMMENT, filename=rel_path, **self.creds,
                                   url=self._generate_text_fragment_link(repo_url, match.get("sentence"), rel_path)) for
                         match in
                         self._text_keyword_iterator(text_content)])

        return matches


def save_to_file(records: List[FullMatch], source: MatchSource, creds: Credentials, with_matched_text: bool = False):
    base_dir = Path("metadata") / "keywords"
    filename = f'{creds.get_ref(".")}.{source.value}.csv'
    if with_matched_text:
        resulting_filename = base_dir / "full" / filename
    else:
        resulting_filename = base_dir / filename
    os.makedirs(resulting_filename.parent, exist_ok=True)

    pd.DataFrame(records).to_csv(resulting_filename, index=False)


if __name__ == "__main__":
    docs_path = Path(".tmp/docs")
    source_code_path = Path(".tmp/source")

    creds = Credentials(author="scverse", repo="scanpy", version="1.10.2", wiki="scanpy.readthedocs.io/en")
    credential_list = [creds]

    for creds in credential_list:
        print(f"Checking out {creds.get_ref()}")
        try:
            # checkout_tag(creds['author'], creds['repo'], creds['version'])

            append_full_text = False
            parser = KeywordParser(quality_attributes, creds, append_full_text=append_full_text)

            if creds.has_wiki():
                matches_wiki = parser.parse_wiki(str(docs_path / f'{creds.repo_path}/{creds.wiki_dir}'))
                save_to_file(matches_wiki, MatchSource.WIKI, creds, append_full_text)

            matches_code_comments = parser.parse_comments(str(source_code_path / creds.get_ref()))
            save_to_file(matches_code_comments, MatchSource.CODE_COMMENT, creds, append_full_text)

            matches_docs = parser.parse_docs(str(source_code_path / creds.get_ref()))
            save_to_file(matches_docs, MatchSource.DOCS, creds, append_full_text)
        except Exception as e:
            print(f"Error processing {creds.get_ref()}: {str(e)}")
