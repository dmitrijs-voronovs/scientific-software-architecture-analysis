import dataclasses
import os
import shelve
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Iterator, ClassVar, Literal, cast, TypeGuard

import dotenv
import pandas as pd
from github import Github
from github.Issue import Issue
from github.IssueComment import IssueComment
from pandas import DataFrame
from tqdm import tqdm

from extract_quality_attribs_from_docs import Credentials
from services.MongoDBConnection import MongoDBConnection

dotenv.load_dotenv()


type InternalReactionKey = Literal["thumbs_up", "thumbs_down", "laugh", "confused", "heart", "hooray", "rocket", "eyes"]
type ReactionKey = Literal[InternalReactionKey, "+1", "-1"]


@dataclass
class ReactionDTO:
    thumbs_up: int = 0
    thumbs_down: int = 0
    laugh: int = 0
    confused: int = 0
    heart: int = 0
    hooray: int = 0
    rocket: int = 0
    eyes: int = 0

    _name_map: ClassVar[Dict[ReactionKey, InternalReactionKey]] = {
        '+1': 'thumbs_up',
        '-1': 'thumbs_down',
    }

    def _get_key(self, key: ReactionKey) -> InternalReactionKey:
        return self._name_map.get(key, cast(InternalReactionKey, key))

    @classmethod
    def is_reaction_key(cls, key: str) -> TypeGuard[ReactionKey]:
        return key in [*cls.__dict__.keys(), "+1", "-1"]

    def add(self, reaction: str):
        if self.is_reaction_key(reaction):
            key = self._get_key(reaction)
            self.__setattr__(key, self.__getattribute__(key) + 1)


@dataclass
class CommentDTO:
    issued_id: int
    id: int
    html_url: str
    body: str
    user: str
    created_at: datetime
    updated_at: datetime
    reactions: ReactionDTO


@dataclass
class IssueDTO:
    id: int
    html_url: str
    number: int
    title: str
    body: str
    state: str
    created_at: datetime
    updated_at: datetime
    closed_at: datetime
    labels: List[str]
    author: str
    assignees: List[str]
    milestone: str
    comments_count: int
    comments_data: List[CommentDTO]
    reactions: ReactionDTO


class GitHubDataFetcher:
    def __init__(self, token: str):
        """
        Initialize the fetcher with GitHub token

        Args:
            token (str): GitHub Personal Access Token
        """
        self.github = Github(token)

    def get_all_issues(self, creds: Credentials, batch_size: int = 10) -> Iterator[List[IssueDTO]]:
        repo = self.github.get_repo(creds.get_repo_path)

        os.makedirs(".cache/issues", exist_ok=True)
        with shelve.open(f".cache/issues/{creds.get_repo_name}") as db:
            since = db.get("since", None)
            # Get all issues (including pull requests)
            issues = repo.get_issues(state='all', direction='asc', since=since) if since else repo.get_issues(state='all', direction='asc')
            total_count = issues.totalCount

            batch = []
            for issue in tqdm(issues, total=total_count, desc="Fetching issues"):
                try:
                    batch.append(self._map_issue_to_dto(issue))

                    if len(batch) == batch_size:
                        tqdm.write(f"Yielding batch of {len(batch)} issues")
                        print(f"Yielding batch of {len(batch)} issues")
                        yield batch
                        db["since"] = issue.created_at
                        batch.clear()

                    # Handle rate limiting
                    if self.github.get_rate_limit().core.remaining < 100:
                        time.sleep(10)

                except Exception as e:
                    print(f"Error processing issue #{issue.number}: {str(e)}")
                    continue

    def _map_issue_to_dto(self, issue: Issue) -> IssueDTO:
        # Get reactions for the issue
        reactions = self._get_reactions(issue)
        # Get comments with their reactions
        comments_data = self._get_comments(issue)
        issue_data = IssueDTO(
            id= issue.id,
            html_url= issue.html_url,
            number= issue.number,
            title= issue.title,
            body= issue.body,
            state= issue.state,
            created_at= issue.created_at,
            updated_at= issue.updated_at,
            closed_at= issue.closed_at,
            labels= [label.name for label in issue.labels],
            author= issue.user.login if issue.user else None,
            assignees= [assignee.login for assignee in issue.assignees],
            milestone= issue.milestone.title if issue.milestone else None,
            comments_count= issue.comments,
            comments_data= comments_data,
            reactions= reactions)
        return issue_data

    def _get_comments(self, issue: Issue) -> List[CommentDTO]:
        """Fetch all comments for an issue with their reactions"""
        comments_data = []

        for comment in issue.get_comments():
            try:
                reactions = self._get_reactions(comment)

                comment_data = CommentDTO(
                    issued_id= issue.id,
                    id= comment.id,
                    html_url= comment.html_url,
                    body= comment.body,
                    user= comment.user.login if comment.user else None,
                    created_at= comment.created_at,
                    updated_at= comment.updated_at,
                    reactions= reactions
                )
                comments_data.append(comment_data)

            except Exception as e:
                print(f"Error processing comment {comment.id}: {str(e)}")
                continue

        return comments_data

    def _get_reactions(self, item: Issue | IssueComment) -> ReactionDTO:
        """Get reaction counts for an issue or comment"""
        reaction_counts = ReactionDTO()

        try:
            reactions = item.get_reactions()
            for reaction in reactions:
                reaction_counts.add(reaction.content)
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


def extract_keywords(path: str):
    pass


class DB:
    @staticmethod
    def insert_issues(documents: List[IssueDTO], creds: Credentials):
        client = MongoDBConnection().get_client()
        res = client['git_issues'][creds.get_repo_name].insert_many([dataclasses.asdict(issue) for issue in documents])
        print(res)


def main():
    creds = Credentials(author="scverse", repo="scanpy", version="1.10.2")
    github_metadata_path = Path(".tmp/metadata")
    metadata_path = github_metadata_path / creds.get_ref()
    issues_path = metadata_path / "issues"
    releases_path = metadata_path / "releases"

    token = os.getenv('GITHUB_TOKEN')
    fetcher = GitHubDataFetcher(token)

    print("Fetching issues...")
    for issues in fetcher.get_all_issues(creds, 2):
        DB.insert_issues(issues, creds)

    DataFrame(pd.read_hdf(f"{issues_path}.h5", key="issues")).to_csv(f"{issues_path}.csv", index=False)
    DataFrame(pd.read_hdf(f"{issues_path}.h5", key="comments")).to_csv(f"{issues_path}.comments.csv", index=False)

    # print("Fetching releases...")
    # fetcher.get_releases(creds)

    print("Done!")




if __name__ == "__main__":
    main()