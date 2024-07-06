import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def get_repos_by_topic(topic, limit=20):
    # Print first 4 characters of token for verification
    print(f"Using token: {GITHUB_TOKEN[:4]}...")

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    params = {
        "q": f"topic:{topic}",
        "sort": "stars",
        "order": "desc",
        "per_page": limit
    }

    response = requests.get(API_URL, headers=headers, params=params)

    if response.status_code == 200:
        repos = response.json()["items"]
        return [
            {
                "name": repo["full_name"],
                "url": repo["html_url"],
                "stars": repo["stargazers_count"],
                "description": repo["description"],
                "language": repo["language"],
                "last_update": repo["updated_at"],
                "created_at": repo["created_at"],
                "forks": repo["forks_count"],
                "open_issues": repo["open_issues_count"],
                "license": repo["license"]["name"] if repo["license"] else "Not specified",
                "topics": repo["topics"]
            }
            for repo in repos
        ]
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return []


def filter_repos(repos, min_stars=100, max_days_since_update=365):
    today = datetime.now()
    return [
        repo for repo in repos
        if repo["stars"] >= min_stars
        and (today - datetime.strptime(repo["last_update"], "%Y-%m-%dT%H:%M:%SZ")).days <= max_days_since_update
    ]


def main():
    topic = input("Enter the topic you want to search for: ")
    all_repos = get_repos_by_topic(topic)

    if not all_repos:
        print("No repositories found or an error occurred.")
        return

    filtered_repos = filter_repos(all_repos)

    print(f"\nTop repositories for topic '{topic}' (filtered):\n")
    # Display top 10 after filtering
    for i, repo in enumerate(filtered_repos[:10], 1):
        print(f"{i}. {repo['name']}")
        print(f"   Stars: {repo['stars']}")
        print(f"   Language: {repo['language']}")
        print(f"   Last updated: {repo['last_update']}")
        print(f"   Created: {repo['created_at']}")
        print(f"   Forks: {repo['forks']}")
        print(f"   Open Issues: {repo['open_issues']}")
        print(f"   License: {repo['license']}")
        print(f"   URL: {repo['url']}")
        print(f"   Description: {repo['description']}")
        print(f"   Topics: {', '.join(repo['topics'])}")
        print()


if __name__ == "__main__":
    main()
