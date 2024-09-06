from pymongo import MongoClient
import subprocess
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime


def run_git_command(repo_path, command):
    return subprocess.check_output(command, cwd=repo_path, universal_newlines=True)


def analyze_code_churn(repo_path, num_commits=100):
    churn_data = defaultdict(int)
    times_modified = defaultdict(int)

    # Get the last 'num_commits' commit hashes
    commit_hashes = run_git_command(repo_path, ['git', 'log', f'-{num_commits}', '--format=%H']).splitlines()

    for i in range(len(commit_hashes) - 1):
        current_commit = commit_hashes[i]
        previous_commit = commit_hashes[i + 1]

        # Get the diff stats between commits
        diff_stats = run_git_command(repo_path, ['git', 'diff', '--numstat', previous_commit, current_commit])

        for line in diff_stats.splitlines():
            if line.strip():
                added, deleted, file_path = line.split('\t')
                if added != '-' and deleted != '-':
                    churn_data[file_path] += int(added) + int(deleted)
                    times_modified[file_path] += 1

    return churn_data, times_modified


def create_churn_graph(churn_data, times_modified):
    G = nx.Graph()

    # Add nodes and edges
    for file, churn in churn_data.items():
        G.add_node(file, churn=churn, modified=times_modified[file])

        # Connect files in the same directory
        dir_name = "/".join(file.split("/")[:-1])
        for other_file in churn_data:
            if "/".join(other_file.split("/")[:-1]) == dir_name and file != other_file:
                G.add_edge(file, other_file)

    return G


def visualize_churn_graph(G):
    print(G)
    pos = nx.spring_layout(G)
    node_sizes = [G.nodes[node]['churn'] * 10 for node in G.nodes()]  # Scale node sizes for visibility
    nx.draw(G, pos, node_size=node_sizes,
            with_labels=True, font_size=8, node_color='lightblue')
    plt.title("Code Churn Graph")
    plt.show()
    plt.savefig("churn_graph.png")


def store_churn_data(project_name, version, churn_data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['code_analysis']
    collection = db['code_churn']

    document = {
        'project': project_name,
        'version': version,
        'timestamp': datetime.now(),
        'churn_data': [{'file': file, 'churn': churn} for file, churn in churn_data.items()]
    }

    collection.insert_one(document)


def main():
    repo_path = ".tmp/scverse/scanpy/master"
    churn_data, times_modified = analyze_code_churn(repo_path)
    churn_graph = create_churn_graph(churn_data, times_modified)
    visualize_churn_graph(churn_graph)

    # Store the churn data in MongoDB
    project_name = "my_project"
    version = "v1.0.0"
    # store_churn_data(project_name, version, churn_data)


if __name__ == "__main__":
    main()
