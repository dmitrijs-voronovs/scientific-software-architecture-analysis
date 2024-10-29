from pathlib import Path
import re

def update_query(query: str):
    query = re.sub(r'db.getCollection\(".+"\).aggregate\(', '', query)
    query = re.sub(r'^\]\)$', ']', query)
    query = re.sub(r'(?<!")\$\w+', r'"\g<0>"', query)
    query = re.sub(r'(?<=regex:) ?/(.+?)/\w*', r'\1', query)
    query = re.sub(r'(?<=\s)(?<!")\$?\w{2,}(?!=")', r'"\g<0>"', query)
    query = re.sub(r'"(true|false)"', r'\1', query)
    return query


def main():
    queries = Path("queries/mongo").glob("*.js")
    for query in queries:
        with open(query, "r") as f:
            content = f.read()
        with open(query.with_suffix(".formatted"), "w") as f:
            f.write(update_query(content))

if __name__ == "__main__":
    main()