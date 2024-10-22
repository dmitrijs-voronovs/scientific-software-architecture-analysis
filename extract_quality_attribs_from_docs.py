import os
import re
from typing import List, Dict

from bs4 import BeautifulSoup

NestedDictType = Dict[str, Dict[str, List[str]]]


def analyze_file(docs: str) -> NestedDictType:
    sentences = re.split(r'[\n.]', docs)
    reverse_index = {}
    # for attr, words in quality_attributes.items():
    for attr, words in quality_attributes.items():
        reverse_index[attr] = {}
        for word in words:
            for sentence in sentences:
                pattern = re.compile(rf'\b{word}')
                match = re.search(pattern, sentence)
                if match:
                    if word in reverse_index[attr]:
                        reverse_index[attr][word].append(sentence)
                    else:
                        reverse_index[attr][word] = [sentence]
    return reverse_index


def strip_html_tags(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def analyze_docs(path: str) -> Dict[str, NestedDictType]:
    matches = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".html"):
                documentation_raw = open(os.path.join(root, file), "r", encoding="utf-8").read()
                text_content = strip_html_tags(documentation_raw)
                doc_analysis = analyze_file(text_content)
                matches[file] = doc_analysis

    return matches


quality_attributes = {
    'sample': ['perf', "optimiz", "speed", "fast"]
}

# Example usage
if __name__ == "__main__":
    matches = analyze_docs(".tmp/docs/Scanpy/scanpy.readthedocs.io/en")
    print(matches)

