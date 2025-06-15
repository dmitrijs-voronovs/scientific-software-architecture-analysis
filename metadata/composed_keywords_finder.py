import re
from collections import defaultdict


def main():
    raw_words = {
        "plug-and-play",
        "legacy system",
        "loose coupl",
        "glue code",
        "federat",
        "composab",
        "B2B",
        "web service",
        "publish-subscribe",
        "abstraction layer",
        "data model",
        "harmoniz",
        "architectur",
        "compon",
        "cot",
        "interoperab",
        "format",
        "integrat",
        "semant",
        "interfac",
        "messag",
        "specif",
        "contract",
        "exchang",
        "protocol",
        "distribut",
        "standard",
        "orb",
        "connector",
        "transform",
        "api",
        "heterogen",
        "ontolog",
        "platform",
        "middlewar",
        "xml",
        "bind",
        "syntact",
        "corba",
        "adapt",
        "translat",
        "mediat",
        "share",
        "rpc",
        "bridg",
        "plug",
        "coordinat",
        "idl",
        "repositori",
        "mismatch",
        "event-bas",
        "compatib",
        "wrapper",
        "soap",
        "cooperat",
        "stub",
        "proxi",
        "ba",
        "incompatib",
        "convers",
        "socket",
        "discoveri",
        "eai",
        "skeleton",
        "marshal",
        "client-serv",
        "plug-and-play",
        "conflict",
        "comol",
        "registri",
        "gateway",
        "interop",
        "controlstructur",
        "controltopolog",
        "peer-to-p",
        "systems-of-system",
        "cot compon",
        "cots-bas system",
        "licens use",
        "model transform",
        "compon base",
        "matur model",
    }

    words_to_process = raw_words.copy()

    composed_words, individual_words = group_keywords(words_to_process)

    composed_matched_substr = get_composed_matches_substr(composed_words, individual_words)
    composed_matched = get_composed_matches(composed_words, individual_words)

    print(composed_matched_substr)
    print(composed_matched)


def group_keywords(words_to_process):
    individual_words = set()
    for word in words_to_process:
        if not " " in word:
            individual_words.add(word)
    composed_words = words_to_process - individual_words
    return composed_words, individual_words


def all_substrings(word):
    for i in range(len(word) + 1):
        yield word[:i]


def get_composed_matches(composed_words, individual_words):
    composed_matched = defaultdict(set)
    for composition in composed_words:
        for word in re.split(r"[ -]", composition):
            if word in individual_words:
                composed_matched[composition].add(word)
    return composed_matched

def get_composed_matches_substr(composed_words, individual_words):
    composed_matched = defaultdict(set)
    for composition in composed_words:
        for word in re.split(r"[ -]", composition):
            for word_combo in all_substrings(word):
                if word_combo in individual_words:
                    if word_combo != word:
                        composed_matched[composition].add(f"{word_combo} (subst from: `{word}`)")
                    else:
                        composed_matched[composition].add(word_combo)
    return composed_matched


if __name__ == "__main__":
    main()