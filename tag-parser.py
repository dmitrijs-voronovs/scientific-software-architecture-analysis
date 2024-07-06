import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Get GitHub API token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# GitHub API base URL
API_BASE_URL = "https://api.github.com"


def get_repo_tags(repo_url):
    # Parse the repository owner and name from the URL
    path_parts = urlparse(repo_url).path.strip('/').split('/')
    if len(path_parts) != 2:
        print(f"Invalid repository URL: {repo_url}")
        return None, None

    owner, repo = path_parts

    # Construct the API endpoint URL
    api_url = f"{API_BASE_URL}/repos/{owner}/{repo}"

    # Set up headers with authentication
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Make the API request
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        repo_data = response.json()
        return repo, repo_data.get("topics", [])
    else:
        print(f"Error fetching data for {repo_url}: {response.status_code}")
        print(f"Response: {response.text}")
        return None, None


def main():
    # Array of repository URLs
    repo_urls = [
        "https://github.com/haddocking/haddock3",
        "https://github.com/qupath/qupath",
        "https://github.com/DedalusProject/dedalus",
        "https://github.com/DeepRank/deeprank2",
        "https://github.com/GrainLearning/grainLearning",
        "https://github.com/matchms/matchms",
        "https://github.com/GooglingTheCancerGenome/sv-callers",
        "https://github.com/amusecode/amuse",
        "https://github.com/SCM-NV/qmflows", "https://github.com/mexca/mexca",
        "https://github.com/duqtools/duqtools",
        "https://github.com/GO-Eratosthenes/dhdt"
    ]

    # Dictionary to store repository names and their tags
    repo_tags_map = {}

    # Process each repository URL
    for url in repo_urls:
        repo_name, tags = get_repo_tags(url)
        if repo_name and tags is not None:
            repo_tags_map[repo_name] = tags

    # Print the resulting map
    print("\nRepository Tags Map:")
    for repo, tags in repo_tags_map.items():
        print(f"\n{repo}:")
        if tags:
            for tag in tags:
                print(f"- {tag}")
        else:
            print("No tags found for this repository.")

    # Print all unique tags
    print("\nAll unique tags:")
    all_tags = {tag for tags in repo_tags_map.values() for tag in tags}
    print(all_tags)
    with open("./code/tags.txt", "w", encoding="utf-8") as f:
        f.writelines("\n".join(all_tags))


if __name__ == "__main__":
    main()