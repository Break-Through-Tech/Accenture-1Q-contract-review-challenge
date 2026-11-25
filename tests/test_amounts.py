import pytest

from src.risk.amounts import find_amounts, group_amounts, parse_amount, parse_number


@pytest.mark.parametrize("text", ["$100,000", "$100K", "100 K $", "100000 $", "$100k",
                                  "$ 100,000", "100,000 USD", "100 thousand dollars"])
def test_hundred_thousand_written_every_way(text):
    amount = parse_amount(text)
    assert (amount.value, amount.currency) == (100_000, "USD")


@pytest.mark.parametrize("text", ["$1M", "$1,000,000", "1000000 $", "$1 million", "$1MM",
                                  "US$ 1MM", "$1.0 million", "1.000.000,00 USD",
                                  "USD 1,000,000.00"])
def test_one_million_written_every_way(text):
    amount = parse_amount(text)
    assert (amount.value, amount.currency) == (1_000_000, "USD")


@pytest.mark.parametrize("text, value", [
    ("$5B", 5e9), ("$5bn", 5e9), ("$5 billion", 5e9),
    ("$1.5 million", 1_500_000), ("$2.5M", 2_500_000),
    ("USD 2,500.50", 2500.5), ("$1,000,000.50", 1_000_000.5), ("$0.05", 0.05),
])
def test_abbreviations_and_decimals(text, value):
    assert parse_amount(text).value == value


@pytest.mark.parametrize("text, value, currency", [
    ("1.000.000,00 €", 1_000_000, "EUR"),      # EU thousands dots, decimal comma
    ("€1.000", 1000, "EUR"),                   # single dot + 3 digits: thousands for euros
    ("2,5 M €", 2_500_000, "EUR"),             # EU decimal comma with abbreviation
    ("EUR 3.5 million", 3_500_000, "EUR"),
    ("£250k", 250_000, "GBP"),
    ("$6.745.000", 6_745_000, "USD"),          # several dots can only be thousands
])
def test_regional_formats_and_currencies(text, value, currency):
    amount = parse_amount(text)
    assert (amount.value, amount.currency) == (value, currency)


@pytest.mark.parametrize("text", [
    "Section 1,000", "10 months", "2,000,000 shares", "five hundred dollars",
    "Payment ($000)",                     # table header: "in thousands"
    "Period 1 $ ***",                     # redacted amount
])
def test_non_amounts_are_ignored(text):
    assert find_amounts(text) == []


def test_currency_mark_belongs_to_the_following_number():
    assert [a.text for a in find_amounts("Restaurant # 1 $40,000.00")] == ["$40,000.00"]


def test_spelled_out_cents_are_not_an_amount():
    found = find_amounts("Five Million and 00/100 Dollars ($5,000,000.00)")
    assert [(a.text, a.value) for a in found] == [("$5,000,000.00", 5_000_000)]


def test_abbreviation_must_end_the_word():
    assert parse_amount("$5 months").value == 5            # "m" of "months" is not million


def test_offsets_index_the_text():
    text = "A fee of $1M, i.e. 1.000.000,00 USD, plus $250."
    for amount in find_amounts(text):
        assert text[amount.start:amount.end] == amount.text


def test_grouping_merges_spellings_of_the_same_value():
    groups = group_amounts(find_amounts("$1M or $1,000,000, and $1M again; fee $250."))
    assert [(g["normalized"], g["written_as"]) for g in groups] == [
        ("USD 1,000,000", ["$1M", "$1,000,000"]),
        ("USD 250", ["$250"]),
    ]


@pytest.mark.parametrize("raw, value", [
    ("1,000", 1000), ("1,000,000", 1e6), ("1.000.000", 1e6), ("1,5", 1.5),
    ("1.5", 1.5), ("1.000.000,50", 1_000_000.5), ("1,000,000.50", 1_000_000.5),
    ("1 000", 1000),
])
def test_parse_number(raw, value):
    assert parse_number(raw) == value
