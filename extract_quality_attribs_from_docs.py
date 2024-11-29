import os
import re
import shelve
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from metadata.repo_info.repo_info import credential_list
from model.Credentials import Credentials
from quality_attributes import quality_attributes
from services.ast_extractor import ext_to_lang, code_comments_iterator
from utils.utils import create_logger_path

AttributeDictType = Dict[str, List[str]]


class TextMatch(Dict):
    keyword: str
    matched_word: str
    match_idx: int
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


class FullMatch(TextMatch, Credentials, Dict):
    filename: Optional[str]
    source: MatchSource
    url: str

    @property
    def id(self):
        return f"{self["url"]}:{self["match_idx"]}"


BASE_GITHUB_URL = "https://github.com"


class KeywordParser:
    context_length = 2000

    def __init__(self, attributes: AttributeDictType, creds: Credentials, *, append_full_text: bool = False):
        self.attributes = attributes
        self.creds = creds
        self.append_full_text = append_full_text

    def matched_keyword_iterator(self, text: str) -> Generator[TextMatch, None, None]:
        text = KeywordParser._clean_text(text)
        for quality_attr, keywords in self.attributes.items():
            pattern = KeywordParser.get_keyword_matching_pattern(keywords)
            for match in re.finditer(pattern, text):
                full_match, keyword, match_idx = match.group(), match.group(1), match.start()
                context = KeywordParser.get_match_context(text, match.start(), match.end())
                text_match = TextMatch(quality_attribute=quality_attr, keyword=keyword, matched_word=full_match,
                                       match_idx=match_idx, sentence=context)
                if self.append_full_text:
                    text_match.text = text
                yield text_match

    @staticmethod
    def _clean_text(text: str):
        text = re.sub(r'\.?(?:\r?\n){2,}', ". ", text)
        text = re.sub(r'\r?\n', "; ", text)
        text = re.sub(r' {2,}', " ", text)
        return text

    @staticmethod
    def get_keyword_matching_pattern(keywords):
        return re.compile(rf'\b({"|".join(keywords)})[a-z-]*\b', re.IGNORECASE)

    @staticmethod
    def _strip_html_tags(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    @staticmethod
    def generate_link(base_url: str, page: str = "") -> str:
        full_url = f"{base_url}/{page}" if page else base_url
        return full_url

    def parse_wiki(self, wiki_path: str) -> List[FullMatch]:
        matches = []
        files = Path(wiki_path).glob("**/*.html")
        for file in tqdm(files, desc=f"Parsing wiki {wiki_path}"):
            abs_path = file
            rel_path = os.path.normpath(os.path.relpath(abs_path, start=wiki_path)).replace("\\", "/")
            try:
                documentation_raw = open(abs_path, "r", encoding="utf-8", errors="replace").read()
                text_content = self._strip_html_tags(documentation_raw)
                matches.extend([FullMatch(**match, source=MatchSource.WIKI, filename=rel_path, **self.creds,
                                          url=self.generate_link(self.creds['wiki'], rel_path)) for match in
                                self.matched_keyword_iterator(text_content)])
            except Exception as error:
                logger.error(f"Parse docs failed for {self.creds.get_ref()}, {file=}: {error=}")
        return matches

    @staticmethod
    def get_match_context(text: str, match_start: int, match_end: int) -> str:
        if len(text) < KeywordParser.context_length:
            return text

        sentence_start, sentence_end = KeywordParser._get_match_sentence(text, match_start, match_end)
        sentence_len = sentence_end - sentence_start
        left_context = KeywordParser.context_length - sentence_len
        delta_side = abs(left_context) // 2

        if left_context < 0:
            return KeywordParser._extract_sentence_segment(text, match_end, match_start, sentence_start, sentence_end)

        left_side = sentence_start - delta_side
        right_side = sentence_end + delta_side
        if left_side < 0:
            return text[:KeywordParser.context_length + 1]
        if right_side > len(text):
            return text[-KeywordParser.context_length:]
        return text[left_side: right_side + 1]

    @staticmethod
    def _extract_sentence_segment(text, match_end, match_start, sentence_start, sentence_end):
        match_len = match_end - match_start
        to_fill = KeywordParser.context_length - match_len
        to_fill_side = to_fill // 2
        segment_from_beginning_end, segment_from_end_start = sentence_start + to_fill_side, sentence_end - to_fill_side
        if match_start < segment_from_beginning_end:
            return text[sentence_start: KeywordParser.context_length + 1]
        elif match_end > segment_from_end_start:
            return text[sentence_end - KeywordParser.context_length: sentence_end + 1]
        else:
            return text[match_start - to_fill_side: match_end + to_fill_side + 1]

    @staticmethod
    def _get_match_sentence(text: str, match_start: int, match_end: int) -> Tuple[int, int]:
        sentence_start, sentence_end = match_start, match_end
        while sentence_start >= 0 and text[sentence_start] != '.':
            sentence_start -= 1
        if sentence_start > 0:
            sentence_start += 1
        while sentence_end < len(text) and text[sentence_end] != '.':
            sentence_end += 1

        return sentence_start, sentence_end

    def parse_docs(self, docs_path: str) -> List[FullMatch]:
        repo_url = self.get_github_repo_url(self.creds)
        matches = []
        docs_extensions = [".md", ".rst", ".txt", ".adoc", ".html"]
        for ext in docs_extensions:
            files = Path(docs_path).glob(f"**/*{ext}")
            for file in tqdm(files, desc="Parsing docs"):
                tqdm.write(str(file))
                abs_path = file
                rel_path = os.path.normpath(os.path.relpath(abs_path, start=docs_path)).replace("\\", "/")
                try:
                    documentation_raw = open(abs_path, "r", encoding="utf-8", errors="replace").read()
                    text_content = self._strip_html_tags(documentation_raw) if ext in ".html" else documentation_raw
                    matches.extend([FullMatch(**match, source=MatchSource.DOCS, filename=rel_path, **self.creds,
                                              url=self.generate_link(repo_url, rel_path)) for match in
                                    self.matched_keyword_iterator(text_content)])
                except Exception as error:
                    logger.error(f"Parse docs failed for {self.creds.get_ref()}, {file=}: {error=}")

        return matches

    @staticmethod
    def get_github_repo_url(creds: Credentials) -> str:
        return f"{BASE_GITHUB_URL}/{creds['author']}/{creds['repo']}/tree/{creds['version']}"

    def parse_comments(self, source_code_path: str) -> List[FullMatch]:
        repo_url = self.get_github_repo_url(self.creds)
        matches = []
        for root, dirs, files in tqdm(os.walk(source_code_path), desc="Parsing code comments"):
            tqdm.write(str(root))
            for file in files:
                supported_languages = ext_to_lang.keys()
                if any(file.endswith(ext) for ext in supported_languages):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.normpath(os.path.relpath(abs_path, source_code_path)).replace("\\", "/")
                    try:
                        for text_content in code_comments_iterator(abs_path):
                            matches.extend([FullMatch(**match, source=MatchSource.CODE_COMMENT, filename=rel_path,
                                                      **self.creds, url=self.generate_link(repo_url, rel_path)) for
                                            match in self.matched_keyword_iterator(text_content)])
                    except Exception as error:
                        logger.error(f"Parse code comments failed for {self.creds.get_ref()}, {file=}: {error=}")
        return matches


def save_to_file(records: List[FullMatch], source: MatchSource, creds: Credentials, with_matched_text: bool = False):
    base_dir = Path("metadata") / "keywords"
    filename = f'{creds.get_ref(".")}.{source.value}.csv'
    if with_matched_text:
        resulting_filename = base_dir / "full" / filename
    else:
        resulting_filename = base_dir / filename
    os.makedirs(resulting_filename.parent, exist_ok=True)

    records_with_id = [{"id": record.id, **record} for record in records]
    pd.DataFrame(records_with_id).to_csv(resulting_filename, index=False)


if __name__ == "__main__":
    docs_path = Path(".tmp/docs")
    source_code_path = Path(".tmp/source")

    logger.add(create_logger_path("keyword_extraction"), mode="w")

    # creds = Credentials(author="scverse", repo="scanpy", version="1.10.2", wiki="scanpy.readthedocs.io/en")
    # creds = Credentials({'author': 'root-project', 'repo': 'root', 'version': 'v6-32-06', 'wiki': 'https://root.cern'})
    # creds = Credentials(
    #     {'author': 'allenai', 'repo': 'scispacy', 'version': 'v0.5.5', 'wiki': 'https://allenai.github.io/scispacy/'})

    # credential_list2 = [creds]

    with shelve.open(".cache/keyword_extraction") as db:
        last_processed = db.get("last_processed", None)
        for creds in credential_list:
            if creds.get_ref() == last_processed:
                logger.info(f"Skipping {creds.get_ref()}")
                continue

            logger.info(f"Processing {creds.get_ref()}")
            try:
                # checkout_tag(creds['author'], creds['repo'], creds['version'])

                append_full_text = False
                parser = KeywordParser(quality_attributes, creds, append_full_text=append_full_text)

                if creds.has_wiki():
                    matches_wiki = parser.parse_wiki(str(docs_path / creds.wiki_dir))
                    save_to_file(matches_wiki, MatchSource.WIKI, creds, append_full_text)

                matches_code_comments = parser.parse_comments(str(source_code_path / creds.get_ref()))
                save_to_file(matches_code_comments, MatchSource.CODE_COMMENT, creds, append_full_text)

                matches_docs = parser.parse_docs(str(source_code_path / creds.get_ref()))
                save_to_file(matches_docs, MatchSource.DOCS, creds, append_full_text)
            except Exception as e:
                logger.error(f"Error processing {creds.get_ref()}: {str(e)}")
            finally:
                db["last_processed"] = creds.get_ref()
