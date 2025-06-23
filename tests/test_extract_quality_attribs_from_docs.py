from processing_pipeline.keyword_matching.extract_quality_attribs_from_docs import KeywordParser


def test_keyword_extraction_patter():
    assert KeywordParser.get_keyword_matching_pattern(["b1", "b2", "a2", "a1", "a_longer"]).pattern == '\\b(a_longer|a2|a1|b1|b2)[a-z-]*\\b'
    assert KeywordParser.get_keyword_matching_pattern(["bunn", "other", "cot compon"]).pattern == '\\b(bunn|cot[a-z-]*\\b \\bcompon|other)[a-z-]*\\b'
