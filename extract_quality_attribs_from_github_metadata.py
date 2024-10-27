import os
import shelve
from pathlib import Path

from extract_quality_attribs_from_docs import Credentials, save_to_file, MatchSource
import pandas as pd
import time


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




if __name__ == "__main__":
    main()