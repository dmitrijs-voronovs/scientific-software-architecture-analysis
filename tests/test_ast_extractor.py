from pathlib import Path

import pytest

from services.ast_extractor import code_comments_iterator


def test_code_comments_iterator_cpp():
    comments = list(code_comments_iterator(get_fixture_path("make_examples_native.cc")))
    assert """/*
 * Copyright 2024 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met: ...
 */""" in comments
    assert '// Initialize samples.' in comments
    assert """// Initialize reference reader.
// We should always have reference name set up, except for some unit tests.
// This code will fail if reference is not set up or cannot be loaded.""" in comments
    # debug_comments(comments)


def debug_comments(comments):
    [print(f"\n{i=:3d}: {comment}") for i, comment in enumerate(comments)]


def get_fixture_path(name):
    return Path("./fixtures/code_samples") / name
