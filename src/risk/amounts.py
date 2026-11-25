"""Find money amounts in contract text and normalize them to one number.

The same amount is written many ways, and all of these read as 1,000,000 USD:

    $1M   $1,000,000   1000000 $   $1 million   US$ 1MM   1.000.000,00 USD

    from src.risk.amounts import find_amounts, parse_amount
    parse_amount("100 K $")            # Amount(value=100000.0, currency='USD', ...)
    find_amounts("a fee of $1M, i.e. $1,000,000")   # two matches, same value

Handles:
* K / M / MM / B abbreviations and thousand / million / billion, with or without a space
* the currency before or after the number ($, US$, USD, EUR, euros, GBP, ...)
* numbers with or without thousands separators
* regional formats: 1.000.000,00 (EU) as well as 1,000,000.00 (US)

Only amounts with a currency marker are matched, so section numbers, dates, share
counts and durations ("10 months") are left alone.

This never rewrites the contract text: the model and every character offset in
the results DB depend on the original text. Amounts are extracted alongside it
(the report lists them per clause, with offsets back into the contract).

Ambiguity: "1,000" is read as one thousand and "1.5" as one and a half. A single
dot followed by exactly three digits ("1.000") is read as thousands only for
euro amounts, where that is the usual convention.
"""

import re
from typing import List, NamedTuple, Optional

_CURRENCY_CODES = {
    "$": "USD", "US$": "USD", "USD": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
    "U.S. DOLLARS": "USD", "US DOLLARS": "USD",
    "€": "EUR", "EUR": "EUR", "EURO": "EUR", "EUROS": "EUR",
    "£": "GBP", "GBP": "GBP",
}
_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}

_CUR_BEFORE = r"US\$|\$|€|£|\b(?:USD|EUR|GBP)\b"
_CUR_AFTER = (r"US\$|\$|€|£|\b(?:USD|EUR|GBP)\b"
              r"|\b(?:U\.S\.\s|US\s)?[Dd]ollars?\b|\b[Ee]uros?\b|\bDOLLARS?\b|\bEUROS?\b")
# Grouped thousands first (1,000,000.00 / 1.000.000,00), else plain (1000000 / 1.5 / 2,5).
_NUMBER = r"\d{1,3}(?:[,.  ']\d{3})+(?:[.,]\d{1,2})?(?!\d)|\d+(?:[.,]\d+)?"
_SUFFIX = r"\s?(?:thousand|million|billion|[Mm][Mm]|mn|bn|[KkMmBb])(?![A-Za-z])"

_PATTERN = re.compile(
    rf"(?P<pre>{_CUR_BEFORE})\s?(?P<num>{_NUMBER})(?P<suf>{_SUFFIX})?"
    # Currency after the number. Not after "/" ("and 00/100 Dollars" is the cents of a
    # spelled-out amount), and not when the mark is followed by another number
    # ("Restaurant # 1 $40,000": that $ belongs to 40,000) or by a redaction
    # ("Period 1 $ ***": the amount itself was removed).
    rf"|(?<![\w.,/])(?P<num2>{_NUMBER})(?P<suf2>{_SUFFIX})?\s?(?P<post>{_CUR_AFTER})"
    r"(?!\s?[\d*\[])"
)
# "$000" in a table header means "in thousands", not an amount of zero.
_LEADING_ZERO = re.compile(r"0\d")


class Amount(NamedTuple):
    text: str          # as written in the contract
    value: float       # normalized number, e.g. 1000000.0
    currency: str      # USD | EUR | GBP
    start: int         # offsets of `text` in the searched string
    end: int

    @property
    def normalized(self) -> str:
        return f"{self.currency} {format_value(self.value)}"


def format_value(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _currency(marker: str) -> str:
    key = " ".join(marker.split()).upper()
    return _CURRENCY_CODES.get(key, _CURRENCY_CODES.get(key.rstrip("S"), "USD"))


def parse_number(text: str, currency: str = "USD") -> Optional[float]:
    """'1,000,000.50' / '1.000.000,50' / '1000000' / '2,5' -> float."""
    digits = re.sub(r"[  ' ]", ",", text.strip())
    commas, dots = digits.count(","), digits.count(".")
    if commas and dots:
        # Whichever comes last is the decimal mark.
        decimal = "," if digits.rfind(",") > digits.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        digits = digits.replace(thousands, "").replace(decimal, ".")
    elif commas:
        tail = digits.split(",")[-1]
        if commas == 1 and len(tail) in (1, 2):
            digits = digits.replace(",", ".")            # 2,5 -> 2.5 (EU decimal)
        else:
            digits = digits.replace(",", "")             # 1,000 / 1,000,000
    elif dots:
        tail = digits.split(".")[-1]
        if dots > 1 or (currency == "EUR" and len(tail) == 3):
            digits = digits.replace(".", "")             # 1.000.000 (EU thousands)
    try:
        return float(digits)
    except ValueError:
        return None


def find_amounts(text: str) -> List[Amount]:
    """Every currency amount in `text`, in order, with offsets into `text`."""
    found = []
    for match in _PATTERN.finditer(text or ""):
        number = match.group("num") or match.group("num2")
        if _LEADING_ZERO.match(number):
            continue
        suffix = (match.group("suf") or match.group("suf2") or "").strip().lower()
        currency = _currency(match.group("pre") or match.group("post"))
        value = parse_number(number, currency)
        if value is None:
            continue
        value *= _MULTIPLIERS.get(suffix, 1)
        found.append(Amount(match.group(0), round(value, 2), currency,
                            match.start(), match.end()))
    return found


def parse_amount(text: str) -> Optional[Amount]:
    """The first amount in `text`, or None."""
    amounts = find_amounts(text)
    return amounts[0] if amounts else None


def group_amounts(amounts: List[Amount]) -> List[dict]:
    """Collapse amounts with the same normalized value, keeping how each was written."""
    groups = {}
    for amount in amounts:
        key = (amount.currency, amount.value)
        entry = groups.setdefault(key, {"currency": amount.currency, "value": amount.value,
                                        "normalized": amount.normalized, "written_as": []})
        if amount.text not in entry["written_as"]:
            entry["written_as"].append(amount.text)
    return list(groups.values())
