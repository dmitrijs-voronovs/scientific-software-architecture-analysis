import os
from typing import List, Generator

import dotenv
from tqdm import tqdm

from cfg.quality_attributes import QualityAttributesMap, quality_attributes
from cfg.repo_credentials import selected_credentials
from processing_pipeline.keyword_matching.services.MongoDB import MongoDB
from processing_pipeline.keyword_matching.utils.save_to_file import save_matches_to_file
from processing_pipeline.keyword_matching.services.KeywordParser import MatchSource, FullMatch, KeywordParser
from model.Credentials import Credentials

dotenv.load_dotenv()

def extract_matches(creds: Credentials, db_query: MongoDB, quality_attributes_map: QualityAttributesMap) -> Generator[(MatchSource, List[FullMatch]), None, None]:
    source_to_generator_map = {MatchSource.ISSUE_COMMENT: db_query.extract_comments,
                               MatchSource.ISSUE: db_query.extract_issues,
                               MatchSource.RELEASES: db_query.extract_releases}
    keyword_parser = KeywordParser(quality_attributes_map, creds)
    for source, gen in source_to_generator_map.items():
        matches = []
        for match in tqdm(gen(), desc=f"Processing {creds.dotted_ref} / {source.value}"):
            matches.extend([FullMatch.from_text_match(text_match, source=source, repo=creds, url=match["html_url"]) for text_match in
                            keyword_parser.matched_keyword_iterator(match["text"])])
        yield source, matches

def save_matched_keywords(creds: Credentials, db_query: MongoDB, quality_attributes_map: QualityAttributesMap):
    for source, matches in extract_matches(creds, db_query, quality_attributes_map):
        save_matches_to_file(matches, source, creds)

def main():
    for creds in selected_credentials:
        print(f"Parsing github metadata for {creds}")
        db = MongoDB(creds)

        save_matched_keywords(creds, db, quality_attributes)
    print("Done!")


if __name__ == "__main__":
    main()
