from cfg.quality_attributes import qa_sorter, QualityAttributesMap
from cfg.repo_credentials import all_credentials
from processing_pipeline.keyword_matching.extract_quality_attribs_from_docs import KeywordParser


def test_keyword_extraction_patter():
    assert KeywordParser.get_keyword_matching_pattern(sorted(["b1", "b2", r"a2\b", r"a1", "a_longer"], key=qa_sorter)).pattern == '\\b(?:(a_longer)|(a2\\b)|(a1)|(b1)|(b2))[a-z-]*\\b'
    assert KeywordParser.get_keyword_matching_pattern(sorted(["bunn", "other", "cot compon"], key=qa_sorter)).pattern == '\\b(?:(cot[a-z-]*\\b \\bcompon)|(other)|(bunn))[a-z-]*\\b'

def test_matched_keyword_iterator():
    test_qas: QualityAttributesMap = {
        "test": ["categ", r"cat\b"],
        "test2": ["kw1", "kw2", "kw3", "kw4"]
    }

    kp = KeywordParser(test_qas, all_credentials[0])
    iter = kp.matched_keyword_iterator("category cat kw3 kw4")
    match = next(iter)
    assert match["keyword"] == "categ"
    assert match["quality_attribute"] == "test"
    assert match["matched_word"] == r"category"

    match = next(iter)
    assert match["keyword"] == "cat"
    assert match["quality_attribute"] == "test"
    assert match["matched_word"] == r"cat"

    match = next(iter)
    assert match["keyword"] == "kw3"
    assert match["quality_attribute"] == "test2"
    assert match["matched_word"] == "kw3"

    match = next(iter)
    assert match["keyword"] == "kw4"
    assert match["quality_attribute"] == "test2"
    assert match["matched_word"] == "kw4"

