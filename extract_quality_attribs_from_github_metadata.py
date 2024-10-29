import dataclasses
import os
import re
import shelve
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Iterator, ClassVar, Literal, cast, TypeGuard, Generator, TypedDict

import dotenv
from github import Github
from github.Issue import Issue
from github.IssueComment import IssueComment
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from tqdm import tqdm

from extract_quality_attribs_from_docs import Credentials, get_keyword_matching_pattern, \
    FullMatch, MatchSource, save_to_file
from quality_attributes import quality_attributes_sample, QualityAttributesMap, quality_attributes
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
    issue_id: int
    _id: int
    html_url: str
    body: str
    user: str | None
    created_at: datetime
    updated_at: datetime
    reactions: ReactionDTO


@dataclass
class IssueDTO:
    _id: int
    html_url: str
    number: int
    pull_request_html_url: str | None
    title: str
    body: str
    state: str
    created_at: datetime
    updated_at: datetime
    closed_at: datetime
    labels: List[str]
    author: str | None
    assignees: List[str]
    milestone: str | None
    comments_count: int
    comments_data: List[CommentDTO]
    reactions: ReactionDTO

    @property
    def id(self):
        return self._id


@dataclass
class ReleaseDTO:
    _id: int
    html_url: str
    tag_name: str
    title: str
    name: str
    body: str
    created_at: datetime
    published_at: datetime
    draft: bool
    prerelease: bool
    author: str | None
    asset_count: int

    @property
    def id(self):
        return self._id


class GitHubDataFetcher:
    def __init__(self, token: str, creds: Credentials):
        """
        Initialize the fetcher with GitHub token

        Args:
            token (str): GitHub Personal Access Token
        """
        self.github = Github(token)
        self.creds = creds

    def get_issues(self, batch_size: int = 10) -> Iterator[List[IssueDTO]]:
        assert batch_size > 0, "Batch size must be greater than 0"
        repo = self.github.get_repo(self.creds.repo_path)

        os.makedirs(".cache/issues", exist_ok=True)
        with shelve.open(f".cache/issues/{self.creds.repo_name}") as db:
            since = db.get("since", None)
            # Get all issues (including pull requests)
            issues = repo.get_issues(state='all', direction='asc', since=since) if since else repo.get_issues(
                state='all', direction='asc')
            total_count = issues.totalCount

            batch = []
            for issue in tqdm(issues, total=total_count, desc="Fetching issues"):
                try:
                    batch.append(self._map_issue_to_dto(issue))

                    if len(batch) == batch_size:
                        tqdm.write(f"Yielding batch of {len(batch)} issues")
                        yield batch
                        db["since"] = issue.created_at
                        batch.clear()

                    # Handle rate limiting
                    if self.github.get_rate_limit().core.remaining < 100:
                        time.sleep(10)

                except Exception as e:
                    print(f"Error processing issue #{issue.number}: {str(e)}")
                    continue

            if len(batch) > 0:
                tqdm.write(f"Yielding batch of {len(batch)} issues")
                yield batch

    def _map_issue_to_dto(self, issue: Issue) -> IssueDTO:
        # Get reactions for the issue
        reactions = self._get_reactions(issue)
        # Get comments with their reactions
        comments_data = self._get_comments(issue)
        issue_data = IssueDTO(
            _id=issue.id,
            html_url=issue.html_url,
            number=issue.number,
            pull_request_html_url=issue.pull_request.html_url if issue.pull_request else None,
            title=issue.title,
            body=issue.body,
            state=issue.state,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            closed_at=issue.closed_at,
            labels=[label.name for label in issue.labels],
            author=issue.user.login if issue.user else None,
            assignees=[assignee.login for assignee in issue.assignees],
            milestone=issue.milestone.title if issue.milestone else None,
            comments_count=issue.comments,
            comments_data=comments_data,
            reactions=reactions)
        return issue_data

    def _get_comments(self, issue: Issue) -> List[CommentDTO]:
        """Fetch all comments for an issue with their reactions"""
        comments_data = []

        for comment in issue.get_comments():
            try:
                reactions = self._get_reactions(comment)

                comment_data = CommentDTO(
                    _id=comment.id,
                    issue_id=issue.id,
                    html_url=comment.html_url,
                    body=comment.body,
                    user=comment.user.login if comment.user else None,
                    created_at=comment.created_at,
                    updated_at=comment.updated_at,
                    reactions=reactions
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

    def get_releases(self, batch_size=10) -> Iterator[List[ReleaseDTO]]:
        assert batch_size > 0, "Batch size must be greater than 0"

        owner, repo_name = self.creds.get("author"), self.creds.get("repo")
        repo = self.github.get_repo(f"{owner}/{repo_name}")

        releases = repo.get_releases()
        total_releases = releases.totalCount

        batch = []
        for release in tqdm(releases, total=total_releases, desc="Fetching releases"):
            try:
                release_data = ReleaseDTO(
                    _id=release.id,
                    html_url=release.html_url,
                    title=release.title,
                    tag_name=release.tag_name,
                    name=release.title,
                    body=release.body,
                    created_at=release.created_at,
                    published_at=release.published_at,
                    draft=release.draft,
                    prerelease=release.prerelease,
                    author=release.author.login if release.author else None,
                    asset_count=release.get_assets().totalCount
                )
                batch.append(release_data)
                if len(batch) == batch_size:
                    tqdm.write(f"Yielding batch of {len(batch)} releases")
                    yield batch
                    batch.clear()

            except Exception as e:
                print(f"Error processing release {release.title}: {str(e)}")
                continue

        if len(batch) > 0:
            tqdm.write(f"Yielding batch of {len(batch)} releases")
            yield batch


def extract_keywords(path: str):
    pass


class MongoTextMatch(TypedDict):
    captures: List[str]
    idx: int
    match: str


class MongoMatch(TypedDict):
    _id: str
    text: str
    html_url: str
    text_match: MongoTextMatch

class DB:
    def __init__(self, creds: Credentials):
        self.creds = creds
        self.client = MongoDBConnection().get_client()

    def _issue_collection(self) -> Collection:
        return self.client['git_issues'][self.creds.repo_name]

    def _releases_collection(self) -> Collection:
        return self.client['git_releases'][self.creds.repo_name]

    def insert_issues(self, documents: List[IssueDTO]):
        table = self._issue_collection()
        try:
            res = table.bulk_write(
                [UpdateOne({"_id": issue.id}, {"$set": dataclasses.asdict(issue)}, upsert=True) for issue in documents])
            print(res)
        except Exception as e:
            print(e)

    def insert_releases(self, documents: List[ReleaseDTO]):
        table = self._releases_collection()
        try:
            res = table.bulk_write(
                [UpdateOne({"_id": release.id}, {"$set": dataclasses.asdict(release)}, upsert=True) for release in
                 documents])
            print(res)
        except Exception as e:
            print(e)

    def extract_comment_body_keywords(self, pattern: re.Pattern) -> CommandCursor[MongoMatch]:
        return self._issue_collection().aggregate([
            {
                "$unwind": "$comments_data"
            },
            {
                "$addFields": {
                    "text": "$comments_data.body",
                    "text_match": {
                        "$regexFindAll": {"input": "$comments_data.body", "regex": pattern}},
                }
            },
            {
                "$match": {
                    "text_match.match": {"$exists": True}
                }
            },
            {
                "$unwind": "$text_match"
            },
            {
                "$project": {
                    "text": 1,
                    "html_url": "$comments_data.html_url",
                    "text_match": 1
                }
            }
        ])

    def extract_issue_body_keywords(self, pattern: re.Pattern) -> CommandCursor[MongoMatch]:
        return self._issue_collection().aggregate([
            {
                "$addFields": {
                    "text": "$body",
                    "text_match": {"$regexFindAll": {"input": "$body", "regex": pattern}},
                }
            },
            {
                "$match": {"text_match.match": {"$exists": True}}
            },
            {
                "$unwind": "$text_match"
            },
            {
                "$project": {
                    "text": 1,
                    "html_url": 1,
                    "text_match": 1
                }
            }
        ]
        )

    def extract_issue_title_keywords(self, pattern: re.Pattern) -> CommandCursor[MongoMatch]:
        return self._issue_collection().aggregate([
            {
                "$addFields": {
                    "text": "$title",
                    "text_match": {"$regexFindAll": {"input": "$title", "regex": pattern}},
                }
            },
            {
                "$match": {"text_match.match": {"$exists": True}}
            },
            {
                "$unwind": "$text_match"
            },
            {
                "$project": {
                    "text": 1,
                    "html_url": 1,
                    "text_match": 1
                }
            }
        ]
        )

    def extract_release_body_keywords(self, pattern: re.Pattern) -> CommandCursor[MongoMatch]:
        return self._releases_collection().aggregate([
            {
                "$addFields": {
                    "text": "$body",
                    "text_match": {"$regexFindAll": {"input": "$body", "regex": pattern}},
                }
            },
            {
                "$match": {"text_match.match": {"$exists": True}}
            },
            {
                "$unwind": "$text_match"
            },
            {
                "$project": {
                    "text": 1,
                    "html_url": 1,
                    "text_match": 1
                }
            }
        ])


def find_sentence_boundaries(text, match_idx):
    sentence_endings = re.compile(r'[.!?\n]')

    # Find the start of the sentence
    start = match_idx
    while start > 0 and not sentence_endings.match(text[start - 1]):
        start -= 1
    # Adjust start to skip past any punctuation or line break
    if start > 0 and text[start - 1] != '\n':
        start += 1  # Skip past the punctuation mark if it isn't a line break

    # Find the end of the sentence
    end = match_idx
    while end < len(text) and not sentence_endings.match(text[end]):
        end += 1
    # Adjust end to include the punctuation or line break
    if end < len(text):
        end += 1  # Include the punctuation mark or line break

    # Extract the sentence and remove extra whitespace
    sentence = text[start:end].strip()
    return sentence


def save_matched_keywords(creds, db, quality_attributes_map: QualityAttributesMap):
    generators_to_sources = {
        db.extract_comment_body_keywords: MatchSource.ISSUE_COMMENT,
        db.extract_issue_body_keywords: MatchSource.ISSUE,
        db.extract_issue_title_keywords: MatchSource.ISSUE,
        db.extract_release_body_keywords: MatchSource.RELEASES
    }
    for quality_attr, keywords in quality_attributes_map.items():
        pattern = get_keyword_matching_pattern(keywords)
        for gen, source in generators_to_sources.items():
            matches = []
            for match in tqdm(gen(pattern), desc=f"Processing {quality_attr} / {source.value}"):
                # text_match > : captures[0] is keyword, full match is match
                sentence = find_sentence_boundaries(match["text"], match["text_match"]["idx"])
                matches.append(FullMatch(
                    quality_attribute=quality_attr,
                    keyword=match["text_match"]["captures"][0],
                    matched_word=match["text_match"]["match"],
                    sentence=sentence,
                    source=source,
                    author=creds['author'],
                    repo=creds['repo'],
                    version=creds['version'],
                    url=match["html_url"],
                ))

            save_to_file(matches, source, creds)


def main():
    creds = Credentials(author="scverse", repo="scanpy", version="1.10.2")
    # creds = Credentials(author="allenai", repo="scispacy", version="v0.5.5")
    github_metadata_path = Path(".tmp/metadata")
    metadata_path = github_metadata_path / creds.get_ref()
    issues_path = metadata_path / "issues"
    releases_path = metadata_path / "releases"

    token = os.getenv('GITHUB_TOKEN')
    fetcher = GitHubDataFetcher(token, creds)
    db = DB(creds)

    # print("Fetching issues...")
    # for issues in fetcher.get_issues():
    #     db.insert_issues(issues)

    # print("Fetching releases...")
    # for releases in fetcher.get_releases(20):
    #     db.insert_releases(releases)

    save_matched_keywords(creds, db, quality_attributes)

    print("Done!")


if __name__ == "__main__":
    main()
