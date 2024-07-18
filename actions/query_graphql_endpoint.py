import asyncio
import json
import os
from typing import Callable, Coroutine, Any

import dotenv
import requests
from pymongo import UpdateOne

GITHUB_GRAPHQL_ENDPOINT = "https://api.github.com/graphql"


async def execute_graphql_query(url, token, query, variables=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    data = {"query": query, "variables": variables}

    response = await asyncio.get_event_loop().run_in_executor(None,
                                                              lambda: requests.post(url, headers=headers,
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

    cursor = await get_cursor_from_cache(category)
    has_next = True
    idx = 0

    while has_next:
        # Execute the GraphQL query with variables
        variables = {"topic_name": category, "number_of_repos": n_repos_to_fetch, 'cursor': cursor}
        try:
            result = await execute_graphql_query(url, token, query, variables)
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

        print(f"got repos {category = } {idx = } {topic['repositories']['totalCount'] = } {has_next =} {cursor = } ")
        idx += n_repos_to_fetch
        update_cache(category, cursor)
        await cb(repos)


def update_cache(category, cursor):
    #     make sure to do it with GIL as there are multiple async tasks
    try:
        with open(".cache/cursors.json", "r") as f:
            cursors = json.load(f)
    except FileNotFoundError:
        cursors = {}

    cursors[category] = cursor
    with open(".cache/cursors.json", "w") as f:
        json.dump(cursors, f)


async def get_cursor_from_cache(category: str):
    try:
        with open(".cache/cursors.json", "r") as f:
            cursors = json.load(f)
            return cursors.get(category, None)
    except FileNotFoundError:
        return None


def wrap_with_update_one_operation(data):
    return UpdateOne({'_id': data['_id']}, {'$set': data}, upsert=True)


def normalize_repo(repo, category):
    keys_with_total_objects = ["watchers", "contributors", "branches", "tags", "deployments", "issuesOpen",
                               "issuesClosed", "issuesAll", "PRsOpen", "PRsClosed", "PRsMerged", "PRsAll", "languages",
                               "releases", ]
    keys_to_omit = keys_with_total_objects + ["owner", "repositoryTopics"]
    return {'_id': repo['resourcePath'], "topic": category, **extract_total_count(repo, keys_with_total_objects),
            "owner": repo['owner']['login'], "languages": repo['languages']['nodes'],
            "repositoryTopics": repo['repositoryTopics']['nodes'], }


def extract_total_count(repo, field_names: list[str]):
    return {f"{field_name}N": repo[field_name]['totalCount'] for field_name in field_names}
