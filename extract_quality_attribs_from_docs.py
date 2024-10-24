import os
import re
import string
from enum import Enum
from typing import List, Dict, Generator

from bs4 import BeautifulSoup

AttributeDictType = Dict[str, List[str]]


class TextMatch(dict):
    pattern: str
    word_match: str
    sentence: str


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


def text_keyword_iterator(text: str, attributes: AttributeDictType) -> Generator[TextMatch, None, None]:
    sentences = re.split(r'[\n.]', text)
    for attr, words in attributes.items():
        for word in words:
            for sentence in sentences:
                pattern = re.compile(rf'\b{word}.*?(?=[\s{re.escape(string.punctuation)}]|$)')
                match = re.search(pattern, sentence)
                if match:
                    yield TextMatch(pattern=word, word_match=match.group(), sentence=sentence)


def strip_html_tags(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def analyze_docs(path: str, creds: Credentials) -> List[FullMatch]:
    matches = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".html"):
                abs_path = os.path.join(root, file)
                path_relative_to_docs = os.path.relpath(os.path.normpath(abs_path), start='.tmp/docs')
                documentation_raw = open(abs_path, "r", encoding="utf-8").read()
                text_content = strip_html_tags(documentation_raw)
                matches.extend(
                    [FullMatch(**match, filename=path_relative_to_docs, source=MatchSource.WIKI.value, **creds) for
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

# Example usage
if __name__ == "__main__":
    credentials = Credentials(author="scverse", repo="scanpy", version="latest")
    matches = analyze_docs(".tmp/docs/Scanpy/scanpy.readthedocs.io/en", credentials)
    print(matches)
