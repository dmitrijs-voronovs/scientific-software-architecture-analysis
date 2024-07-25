import asyncio
from asyncio import Semaphore

import dotenv

from actions.db_actions import upsert_collection_async
from fetcher.CategoryTopics import CategoryTopicsFetcher_isOrganization
from constants.db import PROJECTS_COLLECTION_NAME, PROJECTS_DB_NAME
from services.MongoDBConnection import MongoDBConnection
from utils.paths import Paths

dotenv.load_dotenv()

REPOSITORY_COUNT_PER_QUERY = 15
MAX_PARALLEL_TASKS = 6


async def perform(Fetcher):
    tags = await load_tags()
    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    tasks = [query_and_insert(semaphore, Fetcher, topic) for topic in tags]
    await asyncio.gather(*tasks)
    MongoDBConnection().close_connection()


async def query_and_insert(s: 'Semaphore', Fetcher, category):
    async with s:
        await Fetcher.run(category, REPOSITORY_COUNT_PER_QUERY,
                          lambda data: upsert_collection_async(PROJECTS_DB_NAME,
                                                               PROJECTS_COLLECTION_NAME, data))


async def load_tags():
    # repos = get_golden_repos()
    # tags = get_tags(repos)

    with open(Paths.TAGS, "r") as f:
        tags = f.read().splitlines()
    return tags


def main():
    Fetcher = CategoryTopicsFetcher_isOrganization()
    asyncio.run(perform(Fetcher))


if __name__ == "__main__":
    main()
