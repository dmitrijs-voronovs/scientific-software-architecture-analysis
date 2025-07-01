import concurrent.futures
import os
import re
import shelve
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator, Optional

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from cfg.quality_attributes import quality_attributes
from cfg.repo_credentials import selected_credentials
from constants.abs_paths import AbsDirPath
from model.Credentials import Credentials
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
datapoint_count_per_source = defaultdict(int)  # data points before matching


class KeywordParser:
    context_length = 2000

    def __init__(self, QAs: AttributeDictType, creds: Credentials, *, append_full_text: bool = False):
        self.attributes = QAs
        self.creds = creds
        self.append_full_text = append_full_text
        self.qa_patterns = {qa: KeywordParser.get_keyword_matching_pattern(keywords) for qa, keywords in QAs.items()}

    def matched_keyword_iterator(self, text: str) -> Generator[TextMatch, None, None]:
        if not text:
            return
        text = KeywordParser._clean_text(text)
        for quality_attr, keywords in self.attributes.items():
            pattern = self.qa_patterns[quality_attr]
            for match in re.finditer(pattern, text):
                yield self._extract_match_details(match, quality_attr, text)

    def _extract_match_details(self, match, quality_attr, text):
        full_match, match_idx = match.group(), match.start()
        keyword_idx = next(
            (keyword_idx for keyword_idx, group in enumerate(match.groups()) if group is not None), -1)
        keyword = self.attributes[quality_attr][keyword_idx].replace(r'\b', '')
        context = KeywordParser.get_match_context(text, match.start(), match.end())
        text_match = TextMatch(quality_attribute=quality_attr, keyword=keyword, matched_word=full_match,
                               match_idx=match_idx, sentence=context)
        if self.append_full_text:
            text_match.text = text
        return text_match

    @staticmethod
    def _clean_text(text: str):
        text = re.sub(r'([.!?])?(?:\r?\n)+', lambda m: f"{m.group(1)} " if m.group(1) else ". ", text)
        text = re.sub(r' {2,}', " ", text)
        return text.strip()

    @staticmethod
    def get_keyword_matching_pattern(keywords):
        """Expect list of sorted keywords, to be able to identify related keyword based on match group"""
        end_pattern = r'[a-z-]*\b'
        separator = rf"{end_pattern} \b"
        keywords_with_correct_delimiters = [separator.join(k.split(" ")) if " " in k else k for k in keywords]
        keywords_wrapped_in_groups = [f"({k})" for k in keywords_with_correct_delimiters]
        # noinspection RegExpUnnecessaryNonCapturingGroup
        return re.compile(rf'\b(?:{"|".join(keywords_wrapped_in_groups)}){end_pattern}', re.IGNORECASE)

    @staticmethod
    def _strip_html_tags(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    @staticmethod
    def generate_link(base_url: str, page: str = "") -> str:
        full_url = f"{base_url}/{page}" if page else base_url
        return full_url

    def old_parse_wiki(self, wiki_path: str) -> List[FullMatch]:
        global datapoint_count_per_source
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
                datapoint_count_per_source[(self.creds.repo_name, MatchSource.WIKI)] += 1
            except Exception as error:
                logger.error(f"Parse docs failed for {self.creds.get_ref()}, {file=}: {error=}")
        return matches

    def parse_wiki(self, wiki_path: str, max_workers=5) -> List[FullMatch]:
        global datapoint_count_per_source
        matches = []
        files = Path(wiki_path).glob("**/*.html")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_futures = []
            for file in tqdm(files, desc=f"Parsing wiki {wiki_path}"):
                abs_path = file
                all_futures.append(executor.submit(self._parse_wiki_helper, abs_path, datapoint_count_per_source, file, wiki_path))
            for futures in concurrent.futures.as_completed(all_futures):
                matches.extend(futures.result())
        return matches

    def _parse_wiki_helper(self, abs_path, datapoint_count_per_source, file, wiki_path):
        rel_path = os.path.normpath(os.path.relpath(abs_path, start=wiki_path)).replace("\\", "/")
        try:
            documentation_raw = open(abs_path, "r", encoding="utf-8", errors="replace").read()
            text_content = self._strip_html_tags(documentation_raw)
            datapoint_count_per_source[(self.creds.repo_name, MatchSource.WIKI)] += 1
            return [FullMatch(**match, source=MatchSource.WIKI, filename=rel_path, **self.creds,
                                      url=self.generate_link(self.creds['wiki'], rel_path)) for match in
                            self.matched_keyword_iterator(text_content)]
        except Exception as error:
            logger.error(f"Parse docs failed for {self.creds.get_ref()}, {file=}: {error=}")

    @staticmethod
    def get_match_context(text: str, match_start: int, match_end: int) -> str:
        """ Returns a context string of length `KeywordParser.context_length` centered around the match. """
        if len(text) < KeywordParser.context_length:
            return text

        match_len = match_end - match_start
        remaining = KeywordParser.context_length - match_len
        remaining_left = remaining // 2
        remaining_right = remaining - remaining_left

        context_end = match_end + remaining_right
        if context_end > len(text): return text[-KeywordParser.context_length:]

        context_start = match_start - remaining_left
        if context_start < 0: return text[:KeywordParser.context_length]
        return text[context_start: context_end]

    def parse_docs(self, docs_path: str) -> List[FullMatch]:
        global datapoint_count_per_source
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
                    datapoint_count_per_source[(self.creds.repo_name, MatchSource.DOCS)] += 1
                except Exception as error:
                    logger.error(f"Parse docs failed for {self.creds.get_ref()}, {file=}: {error=}")

        return matches

    @staticmethod
    def get_github_repo_url(creds: Credentials) -> str:
        return f"{BASE_GITHUB_URL}/{creds['author']}/{creds['repo']}/tree/{creds['version']}"

    def parse_comments(self, source_code_path: str) -> List[FullMatch]:
        global datapoint_count_per_source
        repo_url = self.get_github_repo_url(self.creds)
        matches = []
        for root, dirs, files in tqdm(os.walk(source_code_path), desc="Parsing code comments"):
            tqdm.write(str(root))
            for file in files:
                supported_language_extensions = ext_to_lang.keys()
                if Path(file).suffix[1:] in supported_language_extensions:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.normpath(os.path.relpath(abs_path, source_code_path)).replace("\\", "/")
                    try:
                        for text_content in code_comments_iterator(abs_path):
                            matches.extend([FullMatch(**match, source=MatchSource.CODE_COMMENT, filename=rel_path,
                                                      **self.creds, url=self.generate_link(repo_url, rel_path)) for
                                            match in self.matched_keyword_iterator(text_content)])
                            datapoint_count_per_source[(self.creds.repo_name, MatchSource.CODE_COMMENT)] += 1
                    except Exception as error:
                        logger.error(
                            f"Parse code comments failed for {self.creds.get_ref()}, {file=}, {rel_path=}: {error=}")
        return matches


def save_to_file(records: List[FullMatch], source: MatchSource, creds: Credentials, with_matched_text: bool = False):
    base_dir = AbsDirPath.KEYWORDS_MATCHING
    filename = f'{creds.get_ref(".")}.{source.value}.parquet'
    if with_matched_text:
        resulting_filename = base_dir / "full" / filename
    else:
        resulting_filename = base_dir / filename
    os.makedirs(resulting_filename.parent, exist_ok=True)

    if len(records) == 0:
        return

    records_with_id = [{"id": record.id, **record, "source":record["source"].value} for record in records]
    pd.DataFrame(records_with_id).to_parquet(resulting_filename, engine='pyarrow', compression='snappy', index=False)


def save_datapoints_per_source_count(run_id: str):
    data_for_df = []
    for (repo_name, match_source), count in datapoint_count_per_source.items():
        data_for_df.append({"repo_name": repo_name, "match_source": match_source.value,
            # Use .value if you want the string representation of the Enum
            "count": count})
    df = pd.DataFrame(data_for_df)
    df_sorted = df.sort_values(by=["repo_name", "match_source"]).reset_index(drop=True)
    df_sorted.to_csv(AbsDirPath.DATA / f"dataset_size/datapoints_per_source_{run_id}.csv", index=False)


def restore_datapoints_per_source_count(run_id: str):
    global datapoint_count_per_source
    file_path = AbsDirPath.DATA / f"dataset_size/datapoints_per_source_{run_id}.csv"

    if not file_path.exists():
        print(f"Warning: File not found at {file_path}. Returning empty defaultdict.")
        return

    try:
        df = pd.read_csv(file_path)

        # Iterate over DataFrame rows and reconstruct the defaultdict
        for index, row in df.iterrows():
            repo_name = row["repo_name"]
            # Convert string back to MatchSource Enum member
            match_source_str = row["match_source"]
            try:
                match_source = MatchSource(match_source_str)
            except ValueError:
                print(
                    f"Warning: Unknown MatchSource '{match_source_str}' encountered for repo '{repo_name}'. Skipping.")
                continue  # Skip this row if Enum conversion fails

            count = int(row["count"])  # Ensure count is an integer

            datapoint_count_per_source[(repo_name, match_source)] = count

        print(f"Data restored from: {file_path}")

    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file {file_path} is empty. Returning empty defaultdict.")
    except Exception as e:
        print(f"An error occurred while restoring data: {e}")


if __name__ == "__main__":
    cache_dir = AbsDirPath.CACHE / "keyword_extraction"
    os.makedirs(cache_dir, exist_ok=True)

    run_id = "24.06.2025"
    cache_path = cache_dir / run_id
    logger.add(create_logger_path(run_id), mode="w")

    restore_datapoints_per_source_count(run_id)

    with shelve.open(cache_path) as db:
        last_processed = db.get("last_processed", None)
        for creds in selected_credentials:
            # if creds.get_ref() == last_processed:
            #     logger.info(f"Skipping {creds.get_ref()}")
            #     continue

            logger.info(f"Processing {creds.get_ref()}")
            try:
                # checkout_tag(creds['author'], creds['repo'], creds['version'])

                append_full_text = False
                parser = KeywordParser(quality_attributes, creds, append_full_text=append_full_text)

                if creds.has_wiki():
                    matches_wiki = parser.parse_wiki(str(AbsDirPath.WIKIS / creds.wiki_dir))
                    save_to_file(matches_wiki, MatchSource.WIKI, creds, append_full_text)

                source_code_path = str(AbsDirPath.SOURCE_CODE / creds.get_ref())
                matches_code_comments = parser.parse_comments(source_code_path)
                save_to_file(matches_code_comments, MatchSource.CODE_COMMENT, creds, append_full_text)

                matches_docs = parser.parse_docs(source_code_path)
                save_to_file(matches_docs, MatchSource.DOCS, creds, append_full_text)
            except Exception as e:
                logger.error(f"Error processing {creds.get_ref()}: {str(e)}")
            finally:
                db["last_processed"] = creds.get_ref()
                save_datapoints_per_source_count(run_id)
