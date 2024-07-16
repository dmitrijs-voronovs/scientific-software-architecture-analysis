from actions.db_actions import upsert_collection
from cfg.constants import PROJECTS_COLLECTION_NAME, PROJECTS_DB_NAME
from services.MongoDBConnection import MongoDBConnection
from tag_parser.tag_parser import get_tags
from utils.utils import get_golden_repos


def main():
    # repos = get_golden_repos()
    # tags = get_tags(repos)
    # print(tags)
    upsert_collection(PROJECTS_DB_NAME, PROJECTS_COLLECTION_NAME)

    MongoDBConnection().close_connection()


if __name__ == "__main__":
    main()
