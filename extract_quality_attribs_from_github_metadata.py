import os
import shelve
from pathlib import Path

from extract_quality_attribs_from_docs import Credentials, save_to_file, MatchSource
import pandas as pd
import time

from github import Github
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
from typing import Dict, List
import time
import dotenv

dotenv.load_dotenv()


class GitHubDataFetcher:
    def __init__(self, token: str):
        """
        Initialize the fetcher with GitHub token

        Args:
            token (str): GitHub Personal Access Token
        """
        self.github = Github(token)

    def get_all_issues(self, creds: Credentials) -> pd.DataFrame:
        owner, repo_name = creds.get("author"), creds.get("repo")
        repo = self.github.get_repo(f"{owner}/{repo_name}")
        issues_data = []

        # Get all issues (including pull requests)
        issues = repo.get_issues(state='all')
        total_count = issues.totalCount

        for issue in tqdm(issues, total=total_count, desc="Fetching issues"):
            try:
                # Skip pull requests if you want only issues
                # if issue.pull_request:
                #     continue

                # Get reactions for the issue
                reactions = self._get_reactions(issue)

                # Get comments with their reactions
                comments_data = self._get_comments(issue)

                issue_data = {
                    'id': issue.id,
                    'html_url': issue.html_url,
                    'number': issue.number,
                    'title': issue.title,
                    'body': issue.body,
                    'state': issue.state,
                    'created_at': issue.created_at,
                    'updated_at': issue.updated_at,
                    'closed_at': issue.closed_at,
                    'labels': [label.name for label in issue.labels],
                    'author': issue.user.login if issue.user else None,
                    'assignees': [assignee.login for assignee in issue.assignees],
                    'milestone': issue.milestone.title if issue.milestone else None,
                    'comments_count': issue.comments,
                    'comments_data': comments_data,
                    'reactions': reactions
                }
                issues_data.append(issue_data)

                # Handle rate limiting
                if self.github.get_rate_limit().core.remaining < 100:
                    time.sleep(10)

            except Exception as e:
                print(f"Error processing issue #{issue.number}: {str(e)}")
                continue

        return pd.DataFrame(issues_data)

    def _get_comments(self, issue) -> List[Dict]:
        """Fetch all comments for an issue with their reactions"""
        comments_data = []

        for comment in issue.get_comments():
            try:
                reactions = self._get_reactions(comment)

                comment_data = {
                    'id': comment.id,
                    'html_url': comment.html_url,
                    'body': comment.body,
                    'user': comment.user.login if comment.user else None,
                    'created_at': comment.created_at,
                    'updated_at': comment.updated_at,
                    'reactions': reactions
                }
                comments_data.append(comment_data)

            except Exception as e:
                print(f"Error processing comment {comment.id}: {str(e)}")
                continue

        return comments_data

    def _get_reactions(self, item) -> Dict:
        """Get reaction counts for an issue or comment"""
        reaction_counts = {
            '+1': 0, '-1': 0, 'laugh': 0, 'confused': 0,
            'heart': 0, 'hooray': 0, 'rocket': 0, 'eyes': 0
        }

        try:
            reactions = item.get_reactions()
            for reaction in reactions:
                reaction_counts[reaction.content] += 1
        except Exception as e:
            print(f"Error getting reactions: {str(e)}")

        return reaction_counts

    def get_releases(self, creds: Credentials) -> pd.DataFrame:
        owner, repo_name = creds.get("author"), creds.get("repo")
        repo = self.github.get_repo(f"{owner}/{repo_name}")
        releases_data = []

        releases = repo.get_releases()
        total_releases = releases.totalCount

        for release in tqdm(releases, total=total_releases, desc="Fetching releases"):
            try:
                release_data = {
                    'id': release.id,
                    'html_url': release.html_url,
                    'tag_name': release.tag_name,
                    'name': release.title,
                    'body': release.body,
                    'created_at': release.created_at,
                    'published_at': release.published_at,
                    'draft': release.draft,
                    'prerelease': release.prerelease,
                    'author': release.author.login if release.author else None,
                    'asset_count': release.get_assets().totalCount
                }
                releases_data.append(release_data)

            except Exception as e:
                print(f"Error processing release {release.title}: {str(e)}")
                continue

        return pd.DataFrame(releases_data)


def query_issues(path: str, creds: Credentials, batch=3):
    os.makedirs(Path(path).parent, exist_ok=True)
    with shelve.open(".cache/issues") as db:
        if db.get("ready", False):
            return
        start_idx = (db.get("index", 0) + 1) // batch * batch
        db["ready"] = False
        items = []
        for i in range(start_idx, 10):
            print(i)
            db["index"] = i
            if (i % batch == 0):
                pd.DataFrame(items).to_hdf(path, key="issues")
                pd.DataFrame(items).to_csv(path + ".csv", mode="a", index=False)
                items = []
            items.append(dict(name="me", id=i))
            time.sleep(1)
        db["ready"] = True
        del db["index"]



def query_releases(path: str, creds: Credentials):
    pass


def extract_keywords(path: str):
    pass


def main():
    creds = Credentials(author="scverse", repo="scanpy", version="1.10.2")
    github_metadata_path = Path(".tmp/metadata")
    metadata_path = github_metadata_path / f'{creds.get("author")}/{creds.get("repo")}/{creds['version']}'
    issues_path = metadata_path / "issues"
    releases_path = metadata_path / "releases"
    query_issues(str(issues_path) + ".h5", creds)
    # query_releases(str(releases_path) + ".h5", creds)
    issue_keywords = extract_keywords(str(issues_path))
    release_keywords = extract_keywords(str(releases_path))
    # save_to_file(issue_keywords, MatchSource.ISSUE, creds)
    # save_to_file(release_keywords, MatchSource.RELEASES, creds)

    # Initialize fetcher with your GitHub token
    token = os.getenv('GITHUB_TOKEN')
    fetcher = GitHubDataFetcher(token)

    # Example repository
    owner = "example_owner"
    repo = "example_repo"

    # Get issues data
    # print("Fetching issues...")
    # issues_df = fetcher.get_all_issues(creds)

    # Get releases data
    print("Fetching releases...")
    releases_df = fetcher.get_releases(creds)

    # Save to files
    print("Saving data...")
    # issues_df.to_parquet('github_issues.parquet')
    releases_df.to_parquet('github_releases.parquet')

    print("Done!")




if __name__ == "__main__":
    main()