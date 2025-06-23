from processing_pipeline.keyword_matching.extract_quality_attribs_from_github_metadata import DB
from cfg.repo_credentials import all_credentials


def main():
    comment_count_map = {}
    for creds in all_credentials:
        print(f"Parsing github metadata for {creds}")
        db = DB(creds)
        comment_count_map[creds.repo_name] = db.count_comments()[0]["totalComments"]

    print(comment_count_map)
    print("Done!")


if __name__ == "__main__":
    main()
