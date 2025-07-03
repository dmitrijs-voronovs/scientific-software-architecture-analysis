import os
import shelve

from loguru import logger

from cfg.quality_attributes import quality_attributes
from cfg.selected_repos import selected_repos
from constants.abs_paths import AbsDirPath
from processing_pipeline.keyword_matching.model.MatchSource import MatchSource
from processing_pipeline.keyword_matching.services.DatasetCounter import DatasetCounter
from processing_pipeline.keyword_matching.services.KeywordExtractor import SourceCodeKeywordExtractor
from processing_pipeline.keyword_matching.utils.save_to_file import save_matches_to_file
from utilities.utils import create_logger_path


def main():
    cache_dir = AbsDirPath.CACHE / "keyword_extraction"
    os.makedirs(cache_dir, exist_ok=True)
    run_id = "03.07.2025_from_docs"
    cache_path = cache_dir / run_id
    logger.add(create_logger_path(run_id), mode="w")
    dataset_counter = DatasetCounter(run_id)
    dataset_counter.restore_datapoints_per_source_count()
    with shelve.open(cache_path) as db:
        last_processed = db.get("last_processed", None)
        for repo in selected_repos:
            # if repo.id == last_processed:
            #     logger.info(f"Skipping {repo.id}")
            #     continue

            logger.info(f"Processing {repo.id}")
            try:
                # checkout_tag(repo['author'], repo['repo'], repo['version'])

                append_full_text = False

                parser = SourceCodeKeywordExtractor(quality_attributes, repo, append_full_text=append_full_text,
                                                    dataset_counter=dataset_counter)
                if repo.has_wiki():
                    matches_wiki = parser.parse_wiki(str(AbsDirPath.WIKIS / repo.wiki_dir))
                    save_matches_to_file(matches_wiki, MatchSource.WIKI, repo, with_matched_text=append_full_text)

                source_code_path = str(AbsDirPath.SOURCE_CODE / repo.id)
                matches_code_comments = parser.parse_comments(source_code_path)
                save_matches_to_file(matches_code_comments, MatchSource.CODE_COMMENT, repo,
                                     with_matched_text=append_full_text)

                matches_docs = parser.parse_docs(source_code_path)
                save_matches_to_file(matches_docs, MatchSource.DOCS, repo, with_matched_text=append_full_text)
            except Exception as e:
                logger.error(f"Error processing {repo.id}: {str(e)}")
            finally:
                db["last_processed"] = repo.id
    dataset_counter.save_datapoints_per_source_count()


if __name__ == "__main__":
    main()
