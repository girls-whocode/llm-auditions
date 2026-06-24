from solution import count_words


def test_count_words_empty():
    assert count_words("") == {}


def test_count_words_case_insensitive():
    assert count_words("Alpha alpha BETA") == {"alpha": 2, "beta": 1}
