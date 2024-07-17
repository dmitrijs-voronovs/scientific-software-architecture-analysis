import asyncio

from actions.db_actions import upsert_collection, upsert_collection_async
from actions.query_graphql_endpoint import query_topics
from cfg.constants import PROJECTS_COLLECTION_NAME, PROJECTS_DB_NAME
from services.MongoDBConnection import MongoDBConnection
from tag_parser.tag_parser import get_tags
from utils.paths import Paths
from utils.utils import get_golden_repos


async def query_and_insert(s, category):
    async with s:
        await query_topics(category, 15,
                           lambda data: upsert_collection_async(PROJECTS_DB_NAME, PROJECTS_COLLECTION_NAME, data))


async def main():
    # repos = get_golden_repos()
    # tags = get_tags(repos)
    with open(Paths.TAGS, "r") as f:
        tags = f.read().splitlines()
    # print(tags)
    tags = tags[:-1]
    semaphore = asyncio.Semaphore(3)
    tasks = [query_and_insert(semaphore, tag) for tag in tags]
    await asyncio.gather(*tasks)
    MongoDBConnection().close_connection()


if __name__ == "__main__":
    asyncio.run(main())
