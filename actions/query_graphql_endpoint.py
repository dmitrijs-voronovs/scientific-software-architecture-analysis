import asyncio
import json
import os

import dotenv
import requests

GITHUB_GRAPHQL_ENDPOINT = "https://api.github.com/graphql"


async def execute_graphql_query(url, token, query, variables=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "query": query,
        "variables": variables
    }

    response = await asyncio.get_event_loop().run_in_executor(
        None, lambda: requests.post(url, headers=headers, data=json.dumps(data))
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"GraphQL query failed with status code {response.status_code}, {response.reason}")


async def query_topics(category: str, n, cb):
    # Load the GraphQL query from the file
    with open("queries/repos_by_topic.gql", "r") as f:
        query = f.read()

    # Set the GraphQL API endpoint and the Bearer token
    url = GITHUB_GRAPHQL_ENDPOINT
    dotenv.load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    cursor = None
    hasNext = True
    idx = 0

    while hasNext:
        # Execute the GraphQL query with variables
        variables = {
            "topic_name": category,
            "number_of_repos": n,
            'cursor': cursor
        }
        result = await execute_graphql_query(url, token, query, variables)
        topic = result['data']['topic']
        topic_name = topic['name']
        page_info = topic['repositories']['pageInfo']
        cursor = page_info['endCursor']
        hasNext = page_info['hasNextPage']

        repos = [{**repo, "topic": topic_name, '_id': repo['resourcePath']} for repo in topic['repositories']['nodes']]

        print(f"got repos {category = } {idx = } {topic['repositories']['totalCount'] = } {hasNext =} ")
        idx += n

        await cb(repos)

# async def main():
#     topics = ["haddock", "pathology"]
#     tasks = [query_topics(topic, 2) for topic in topics]
#     await asyncio.gather(*tasks)
#
# if __name__ == "__main__":
#     asyncio.run(main())
