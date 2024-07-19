import asyncio
import json
import os
from typing import Callable, Coroutine, Any

import dotenv
import requests
from pymongo import UpdateOne

from services.CategoryCache import CategoryCache

GITHUB_GRAPHQL_ENDPOINT = "https://api.github.com/graphql"


async def execute_graphql_query(url, token, query, variables=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    data = {"query": query, "variables": variables}

    response = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.post(url, headers=headers,
                                                                                          data=json.dumps(data)))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"GraphQL query failed with status code {response.status_code}, {response.reason}")


async def query_topics(category: str, n_repos_to_fetch: int, cb: Callable[[object], Coroutine[Any, Any, None]]):
    # Load the GraphQL query from the file
    with open("queries/repos_by_topic.gql", "r") as f:
        query = f.read()

    # Set the GraphQL API endpoint and the Bearer token
    url = GITHUB_GRAPHQL_ENDPOINT
    dotenv.load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    cursor, finished = CategoryCache.get(category, (None, False))
    has_next = not finished
    loop = 0

    while has_next:
        variables = {"topic_name": category, "number_of_repos": n_repos_to_fetch, 'cursor': cursor}
        try:
            result = await execute_graphql_query(url, token, query, variables)
            # print("executed query", result, f"{category = }, {loop = }, {cursor =}")
        except Exception as e:
            print("stumbled upon an exception", e)
            await asyncio.sleep(10)
            continue
        topic = result['data']['topic']
        page_info = topic['repositories']['pageInfo']
        cursor = page_info['endCursor']
        has_next = page_info['hasNextPage']

        repos = [wrap_with_update_one_operation(normalize_repo(repo, category)) for repo in
                 topic['repositories']['nodes']]

        print(
            f"got repos {category = }, {loop = }, idx={loop * n_repos_to_fetch}, {topic['repositories']['totalCount'] = }, {has_next =}, {cursor = } ")
        loop += 1
        CategoryCache.set(category, (cursor, not has_next))
        await cb(repos)

    print(f"Finished fetching {category = }")


def wrap_with_update_one_operation(data):
    return UpdateOne({'_id': data['_id']}, {'$set': data}, upsert=True)


def normalize_repo(repo, category):
    keys_with_total_objects = ["watchers", "contributors", "branches", "tags", "deployments", "issuesOpen",
                               "issuesClosed", "issuesAll", "PRsOpen", "PRsClosed", "PRsMerged", "PRsAll", "languages",
                               "releases", ]
    keys_to_omit = keys_with_total_objects + ["owner", "repositoryTopics"]
    filtered_repo = {k: v for k, v in repo.items() if k not in keys_to_omit}
    return {'_id': repo['resourcePath'], "topic": category, **extract_total_count(repo, keys_with_total_objects),
            **filtered_repo, "owner": repo['owner']['login'], "languages": repo['languages']['nodes'],
            "repositoryTopics": repo['repositoryTopics']['nodes'], }


def extract_total_count(repo, field_names: list[str]):
    return {f"{field_name}N": repo[field_name]['totalCount'] for field_name in field_names}
