from sk_assistant.parsing import is_underline, parse_rst

SAMPLE = """
.. _preprocessing:

==================
Preprocessing data
==================

Intro text...

.. _preprocessing_scaler:

Standardization, or mean removal and variance scaling
=====================================================

Text...
======= ======= =======
column1 column2 column3
======= ======= =======

Scaling features to a range
---------------------------
"""


def test_parse_splits_page_into_sections() -> None:
    parsed_sample = parse_rst(text=SAMPLE, page="preprocessing")
    numbers_of_sections = len(parsed_sample)
    assert numbers_of_sections == 2
    titles_of_sections = [section.title for section in parsed_sample]
    assert titles_of_sections == [
        "Preprocessing data",
        "Standardization, or mean removal and variance scaling",
    ]


def test_page_title() -> None:
    parsed_sample = parse_rst(text=SAMPLE, page="preprocessing")
    assert parsed_sample != []
    for section in parsed_sample:
        assert section.page_title == "Preprocessing data"


def test_no_overline_underline_label() -> None:
    parsed_sample = parse_rst(text=SAMPLE, page="preprocessing")
    assert parsed_sample != []

    for section in parsed_sample:
        text = section.text
        lines = text.splitlines()
        for i in range(len(lines)):
            assert not is_underline(lines[i])
            assert ".. _" not in lines[i]


def test_valid_URL() -> None:
    parsed_sample = parse_rst(text=SAMPLE, page="preprocessing")
    assert parsed_sample != []
    for section in parsed_sample:
        assert section.url == "https://scikit-learn.org/1.9/modules/preprocessing.html"


def test_check_content() -> None:
    parsed_sample = parse_rst(text=SAMPLE, page="preprocessing")
    assert parsed_sample[0].text.strip() == "Intro text..."
    assert parsed_sample[1].text.strip() == "\n".join(
        ["Text...", "======= ======= =======", "column1 column2 column3", "======= ======= ======="]
    )
