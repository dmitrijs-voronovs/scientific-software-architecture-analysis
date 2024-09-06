import os
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import subprocess
from services.git import checkout_tag


def get_repo_tags(author, repo_name):
    repo_url = f"https://github.com/{author}/{repo_name}.git"
    result = subprocess.run(['git', 'ls-remote', '--tags', repo_url], capture_output=True, text=True)
    tags = [line.split('refs/tags/')[-1] for line in result.stdout.splitlines()]
    print(tags)
    return tags


def collect_file_metadata(path):
    file_counts = defaultdict(int)
    line_counts = defaultdict(int)
    file_categories = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file)
            extension = extension.lower()[1:] if extension else 'no_extension'

            # Categorize the file
            if 'test' in file.lower() or 'test' in root.lower():
                category = 'test'
            elif 'doc' in file.lower() or 'doc' in root.lower():
                category = 'doc'
            else:
                category = 'other'

            # Count files
            file_counts[extension] += 1
            file_categories[extension] = category

            # Count lines
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for line in f)
                line_counts[extension] += line_count

    return {
        'file_counts': dict(file_counts),
        'line_counts': dict(line_counts),
        'file_categories': file_categories
    }


def process_repo_tags(author, repo_name, tags):
    all_metadata = []
    for tag in tags:
        print(f"Processing tag: {tag}")
        path = checkout_tag(author, repo_name, tag)
        metadata = collect_file_metadata(path)
        metadata['author'] = author
        metadata['repo'] = repo_name
        metadata['tag'] = tag
        all_metadata.append(metadata)
    return all_metadata


def create_dataframe(all_metadata):
    rows = []
    for metadata in all_metadata:
        for extension, count in metadata['file_counts'].items():
            rows.append({
                'author': metadata['author'],
                'repo': metadata['repo'],
                'tag': metadata['tag'],
                'extension': extension,
                'file_count': count,
                'line_count': metadata['line_counts'][extension],
                'category': metadata['file_categories'][extension]
            })
    return pd.DataFrame(rows)


def save_dataframe(df, author, repo_name):
    os.makedirs('./metadata', exist_ok=True)
    df.to_csv(f'./metadata/{author}_{repo_name}_metadata.csv', index=False)


def create_interactive_plot(df):
    fig = go.Figure()

    for category in ['test', 'doc', 'other']:
        df_category = df[df['category'] == category]

        fig.add_trace(go.Scatter(
            x=df_category['extension'],
            y=df_category['file_count'],
            mode='markers',
            name=f'{category.capitalize()} Files',
            marker=dict(size=10),
            visible=True
        ))

        fig.add_trace(go.Scatter(
            x=df_category['extension'],
            y=df_category['line_count'],
            mode='markers',
            name=f'{category.capitalize()} Lines',
            marker=dict(size=10),
            visible=False
        ))

    # Create and add dropdown
    dropdown_buttons = [
        {'label': "File Counts", 'method': "update",
         'args': [{"visible": [True, False] * 3}, {'yaxis': {'title': 'File Count'}}]},
        {'label': "Line Counts", 'method': "update",
         'args': [{"visible": [False, True] * 3}, {'yaxis': {'title': 'Line Count'}}]}
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Add slider for tags
    steps = []
    for tag in df['tag'].unique():
        step = dict(
            method="update",
            args=[{"visible": [True] * len(fig.data)}],
            label=tag
        )
        for trace in fig.data:
            step["args"][0]["visible"][fig.data.index(trace)] = tag in df[df['extension'].isin(trace.x)]['tag'].values
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Tag: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_title="File Extension",
        yaxis_title="Count",
        title="File and Line Counts by Extension and Category"
    )

    fig.show()


def main():
    author = "scverse"
    repo_name = "scanpy"

    tags = get_repo_tags(author, repo_name)
    all_metadata = process_repo_tags(author, repo_name, tags)

    df = create_dataframe(all_metadata)
    save_dataframe(df, author, repo_name)

    create_interactive_plot(df)


if __name__ == "__main__":
    main()