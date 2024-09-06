import os
from collections import defaultdict
from services.git import checkout_tag


def collect_file_metadata(path):
    file_counts = defaultdict(int)
    line_counts = defaultdict(int)
    test_counts = {'files': 0, 'lines': 0}
    doc_counts = {'files': 0, 'lines': 0}

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file)

            # Categorize the file
            if 'test' in file.lower() or 'test' in root.lower():
                category = 'test'
            elif 'doc' in file.lower() or 'doc' in root.lower():
                category = 'doc'
            else:
                category = extension.lower()[1:] if extension else 'no_extension'

            # Count files
            file_counts[category] += 1

            # Count lines
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for line in f)
                line_counts[category] += line_count

            # Update test and doc counts
            if category == 'test':
                test_counts['files'] += 1
                test_counts['lines'] += line_count
            elif category == 'doc':
                doc_counts['files'] += 1
                doc_counts['lines'] += line_count

    return {
        'file_counts': dict(file_counts),
        'line_counts': dict(line_counts),
        'test_counts': test_counts,
        'doc_counts': doc_counts
    }


def main():
    author = "scverse"
    repo_name = "scanpy"
    tag = "1.10.1"
    path = checkout_tag(author, repo_name, tag)

    metadata = collect_file_metadata(path)

    print(f"Repository: {author}/{repo_name} (tag: {tag})")
    print("\nFile counts by extension:")
    for ext, count in metadata['file_counts'].items():
        print(f"  {ext}: {count}")

    print("\nLine counts by extension:")
    for ext, count in metadata['line_counts'].items():
        print(f"  {ext}: {count}")

    print(f"\nTest files: {metadata['test_counts']['files']}")
    print(f"Test lines: {metadata['test_counts']['lines']}")

    print(f"\nDocumentation files: {metadata['doc_counts']['files']}")
    print(f"Documentation lines: {metadata['doc_counts']['lines']}")


if __name__ == "__main__":
    main()