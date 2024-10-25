import os
import re
import string
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

AttributeDictType = Dict[str, List[str]]


class TextMatch(dict):
    keyword: str
    matched_word: str
    sentence: str
    quality_attribute: str


class MatchSource(Enum):
    WIKI = "WIKI"
    DOCS = "DOCS"
    ISSUE = "ISSUE"
    ISSUE_COMMENT = "ISSUE_COMMENT"
    CODE_COMMENT = "CODE_COMMENT"


class Credentials(Dict):
    author: str
    repo: str
    version: str


class FullMatch(TextMatch, Credentials):
    filename: str
    source: MatchSource
    url: str


def text_keyword_iterator(text: str, attributes: AttributeDictType) -> Generator[TextMatch, None, None]:
    sentences = re.split(r'[\n.]', text)
    for quality_attr, keyword in attributes.items():
        for word in keyword:
            for sentence in sentences:
                # Word begins with the keyword. Ends either at the end of the word, at the punctuation or end of line.
                pattern = re.compile(rf'\b{word}.*?(?=[\s{re.escape(string.punctuation)}]|$)')
                match = re.search(pattern, sentence)
                if match:
                    yield TextMatch(quality_attribute=quality_attr, keyword=word, matched_word=match.group(), sentence=sentence)


def strip_html_tags(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def generate_text_fragment_link(base_url: str, text: str, page: str = "") -> str:
    full_url = f"{base_url}/{page}" if page else base_url
    encoded_text = urllib.parse.quote(text)
    return f"{full_url}#:~:text={encoded_text}"


def parse_wiki(wiki_path: str, creds: Credentials, wiki_url: str) -> List[FullMatch]:
    matches = []
    for root, dirs, files in tqdm(os.walk(wiki_path), desc="Parsing wiki"):
        tqdm.write(f"WIKI parsing > Dir: {root} | Dirs: {len(dirs)} | Files: {len(files)}")
        for file in files:
            if file.endswith(".html"):
                abs_path = os.path.join(root, file)
                rel_path = os.path.normpath(os.path.relpath(abs_path, wiki_path)).replace("\\", "/")
                documentation_raw = open(abs_path, "r", encoding="utf-8").read()
                text_content = strip_html_tags(documentation_raw)
                matches.extend(
                    [FullMatch(**match, source=MatchSource.WIKI.value, filename=rel_path, **creds,
                               url=generate_text_fragment_link(wiki_url, match.get("sentence"), rel_path)) for
                     match in
                     text_keyword_iterator(text_content, quality_attributes_sample)])

    return matches


quality_attributes = {
    "Availability": ["avail", "downtime", "outage", "reliab", "fault", "failure", "error", "robust", "toler",
                     "resilien", "recover", "repair", "failover", "fail-safe", "backup", "redundant", "mask",
                     "degraded", "mainten", "heartbeat", "ping", "echo", "rollback", "checkpoint", "spare", "reboot",
                     "alive", "down", "time"],
    "Deployability": ["deploy", "release", "update", "install", "rollout", "rollback", "upgrade", "integrat",
                      "continuous", "version", "hotfix", "patch", "CI/CD", "pipeline", "configurat", "rolling",
                      "kill switch", "feature toggle", "toggle", "canary", "A/B"],
    "Energy Efficiency": ["energy", "power", "consumption", "efficient", "battery", "charge", "drain", "watt", "joule",
                          "green", "sustainab", "meter", "monitor", "reduce", "allocate", "adapt", "schedul", "sensor",
                          "fusion", "kill", "benchmark"],
    "Integrability": ["integrat", "interoperab", "interface", "depend", "inject", "wrap", "bridg", "mediat", "abstract",
                      "service", "discover", "adapter", "normalize", "standard", "contract", "protocol", "message",
                      "synchronization", "state", "data", "syntactic", "semantic", "publish-subscribe", "service bus",
                      "rout"],
    "Modifiability": ["change", "modify", "adapt", "evolve", "extend", "enhance", "flexible", "maintainab", "refactor",
                      "rewrite", "config", "parameteriz", "polymorphi", "inherit", "coupling", "cohesion",
                      "abstraction", "layers", "sandbox", "scalab", "variab", "portab", "location independence",
                      "plug-in", "microkernel", "dept"],
    "Performance": ["perform", "latency", "throughput", "response time", "miss rate", "load", "scalab", "bottleneck",
                    "tune", "optimiz", "concurren", "multi-thread", "race condition", "queue", "cache", "throttle",
                    "load balanc"],
    "Safety": ["safe", "unsafe", "hazard", "fault", "failure", "risk", "avoid", "detect", "remediat", "redund",
               "predict", "timeout", "sanity check", "abort", "degrad", "mask", "barrier", "firewall", "interlock",
               "recover", "rollback", "repair", "reconfigur"],
    "Security": ["secur", "attack", "intrusion", "denial-of-service", "confidential", "integrity", "authoriz", "access",
                 "firewall", "interlock", "encrypt", "validat", "sanitiz", "audit", "threat", "attack", "expose",
                 "checksum", "hash", "man-in-the-middle", "password", "certificate", "authenticat", "biometric",
                 "CAPTCHA", "injection", "cross-site scripting", "XSS"],
    "Testability": ["test", "control", "sandbox", "assert", "dependency injection", "strategy", "mock", "stub",
                    "complex", "report", "verbose", "logg", "resource monitor", "dependency", "benchmark"],
    "Usability": ["usable", "user-friendly", "user experience", "UX", "efficien", "feedback", "help", "undo", "pause",
                  "resume", "reversible", "correction", "progress bar", "clear", "learn", "responsiv", "simpl", "guid",
                  "intuit"], }

quality_attributes_sample = {
    'sample': ['perf', "optimiz", "speed", "fast"]
}


def save_to_file(records: List, source: MatchSource, creds: Credentials):
    dir = Path("metadata") / "keywords"
    os.makedirs(dir, exist_ok=True)
    filename = f'{creds.get("author")}.{creds.get("repo")}.{creds.get("version")}.{source.value}.csv'
    pd.DataFrame(records).to_csv(dir / filename, index=False)


if __name__ == "__main__":
    creds = Credentials(author="scverse", repo="scanpy", version="latest")
    wiki_url = "scanpy.readthedocs.io/en"
    protocol = "https://"
    docs_path = Path(".tmp/docs")
    matches = parse_wiki(str(docs_path / f'{creds.get("author")}/{creds.get("repo")}/{wiki_url}'), creds,
                         f'{protocol}{wiki_url}')
    save_to_file(matches, MatchSource.WIKI, creds)
    print(matches)
