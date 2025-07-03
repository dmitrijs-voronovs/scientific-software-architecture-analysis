import os

import dotenv
from tqdm import tqdm

from cfg.quality_attributes import QualityAttributesMap, quality_attributes
from cfg.repo_credentials import selected_credentials
from processing_pipeline.keyword_matching.services.GithubDataFetcher import GithubDataFetcher
from processing_pipeline.keyword_matching.services.MongoDB import MongoDB
from processing_pipeline.keyword_matching.utils.save_to_file import save_matches_to_file
from processing_pipeline.keyword_matching.services.KeywordParser import MatchSource, FullMatch, KeywordParser
from model.Credentials import Credentials

dotenv.load_dotenv()

def main():
    token = os.getenv('GITHUB_TOKEN')

    for creds in selected_credentials:
        fetcher = GithubDataFetcher(token, creds)
        db = MongoDB(creds)
        print("Fetching issues...")
        for issues in fetcher.get_issues():
            db.insert_issues(issues)
        print("Fetching releases...")
        for releases in fetcher.get_releases(20):
            db.insert_releases(releases)

    print("Done!")


if __name__ == "__main__":
    main()
