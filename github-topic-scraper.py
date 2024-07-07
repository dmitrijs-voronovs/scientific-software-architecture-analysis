import os
import json
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub GraphQL API endpoint
GITHUB_API_URL = "https://api.github.com/graphql"

# GitHub personal access token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Headers for the API request
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}


def run_query(query, variables):
    """Execute a GraphQL query."""
    request = requests.post(GITHUB_API_URL,
                            json={
                                'query': query,
                                'variables': variables
                            },
                            headers=headers,
                            timeout=5)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(
            f"Query failed with status code: {request.status_code}. {request.json()}"
        )


def get_repositories(topic, num_repos, cursor=None):
    """Fetch repositories for a given topic."""
    query = """
    query($topic: String!, $num_repos: Int!, $cursor: String) {
      search(query: $topic, type: REPOSITORY, first: $num_repos, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          ... on Repository {
            name
            url
            stargazerCount
            forkCount
            description
            watchers {
              totalCount
            }
            releases {
              totalCount
            }
            tags: refs(refPrefix: "refs/tags/", first: 0) {
              totalCount
            }
            defaultBranchRef {
              target {
                ... on Commit {
                  history {
                    totalCount
                  }
                  committedDate
                }
              }
            }
            licenseInfo {
              name
            }
            collaborators {
              totalCount
            }
            object(expression: "HEAD:README.md") {
              ... on Blob {
                text
              }
            }
          }
        }
      }
    }
    """
    variables = {
        "topic": f"topic:{topic}",
        "num_repos": num_repos,
        "cursor": cursor,
    }
    return run_query(query, variables)


def scrape_repos(topic, num_repos, start_cursor=None):
    """Scrape repositories for a given topic."""
    repositories = []
    cursor = start_cursor

    while len(repositories) < num_repos:
        try:
            result = get_repositories(topic,
                                      min(100, num_repos - len(repositories)),
                                      cursor)
            repos = result['data']['search']['nodes']

            for repo in repos:
                repo_data = {
                    "topic":
                    topic,
                    "name":
                    repo['name'],
                    "url":
                    repo['url'],
                    "stars":
                    repo['stargazerCount'],
                    "forks":
                    repo['forkCount'],
                    "description":
                    repo['description'],
                    "watchers":
                    repo['watchers']['totalCount'],
                    "releases":
                    repo['releases']['totalCount'],
                    "tags":
                    repo['tags']['totalCount'],
                    "total_commits":
                    repo['defaultBranchRef']['target']['history']
                    ['totalCount'],
                    "last_commit_date":
                    repo['defaultBranchRef']['target']['committedDate'],
                    "license":
                    repo['licenseInfo']['name']
                    if repo['licenseInfo'] else None,
                    "contributors":
                    repo['collaborators']['totalCount'],
                    "readme":
                    repo['object']['text'] if repo['object'] else None,
                }
                repositories.append(repo_data)

            cursor = result['data']['search']['pageInfo']['endCursor']
            if not result['data']['search']['pageInfo']['hasNextPage']:
                break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            break

    return repositories, cursor


def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_to_csv(data, filename):
    """Save data to a CSV file."""
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def main(topic, num_repos, start_cursor=None):
    """Main function to scrape and save repository data."""
    print(f"Scraping {num_repos} repositories for topic: {topic}")
    repositories, end_cursor = scrape_repos(topic, num_repos, start_cursor)

    if repositories:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{topic}_repos_{timestamp}.json"
        csv_filename = f"{topic}_repos_{timestamp}.csv"

        save_to_json(repositories, json_filename)
        save_to_csv(repositories, csv_filename)

        print(f"Scraped {len(repositories)} repositories.")
        print(f"Data saved to {json_filename} and {csv_filename}")
        print(f"Last cursor: {end_cursor}")
    else:
        print("No repositories scraped. Check your token or try again later.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Scrape GitHub repositories by topic")
    parser.add_argument("topic", help="GitHub topic to scrape")
    parser.add_argument("num_repos",
                        type=int,
                        help="Number of repositories to scrape")
    parser.add_argument("--start_cursor",
                        help="Cursor to start from (for resuming)",
                        default=None)
    args = parser.parse_args()

    main(args.topic, args.num_repos, args.start_cursor)
