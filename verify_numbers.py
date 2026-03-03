"""Number extraction, variants, and matching."""
import re
from typing import List, Optional, Tuple

from verify_constants import METRIC_SYNONYMS, UNITS
from verify_io import normalize_invisibles, unique_keep_order

_YEAR_PAT = re.compile(r"(19|20)\d{2}\s*년?")
_WORD_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}[A-Za-z0-9\-_/]*")


def extract_numbers(text: str) -> List[str]:
    """Extract number-like tokens including commas/decimals/units."""
    text = normalize_invisibles(text)
    pattern = r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s?(?:%|배|개월|년|원|백만원|억원|%p|p))?"
    nums = re.findall(pattern, text)
    nums = [re.sub(r"\s+", "", n) for n in nums]
    nums = [n for n in nums if re.search(r"\d", n)]
    return unique_keep_order(nums)


def numbers_in_claim(text: str) -> List[str]:
    return extract_numbers(text)


def _add_commas_to_int(int_str: str) -> str:
    if not int_str.isdigit():
        return int_str
    s = int_str
    parts: List[str] = []
    while len(s) > 3:
        parts.append(s[-3:])
        s = s[:-3]
    parts.append(s)
    return ",".join(reversed(parts))


def _convert_unit(num_value: float, from_unit: str, to_unit: str) -> Optional[float]:
    from_unit = normalize_invisibles(from_unit).lower()
    to_unit = normalize_invisibles(to_unit).lower()
    unit_factors = {
        "원": 1,
        "천원": 1000,
        "만원": 10000,
        "백만원": 1000000,
        "억원": 100000000,
    }
    from_factor = unit_factors.get(from_unit)
    to_factor = unit_factors.get(to_unit)
    if from_factor is None or to_factor is None:
        return None
    return (num_value * from_factor) / to_factor


def number_variants(token: str) -> List[str]:
    token = normalize_invisibles(token).strip()
    if not token:
        return []
    m = re.match(r"^([0-9][0-9,]*(?:\.[0-9]+)?)(.*)$", token)
    if not m:
        return [token]
    num, unit = m.group(1), m.group(2) or ""
    unit = normalize_invisibles(unit)
    if "." in num:
        int_part, dec_part = num.split(".", 1)
        int_no = int_part.replace(",", "")
        num_no = f"{int_no}.{dec_part}"
        num_comma = f"{_add_commas_to_int(int_no)}.{dec_part}"
        num_value = float(f"{int_no}.{dec_part}")
    else:
        int_no = num.replace(",", "")
        num_no = int_no
        num_comma = _add_commas_to_int(int_no)
        num_value = float(int_no)
    base = [f"{num}{unit}", f"{num_no}{unit}", f"{num_comma}{unit}"]
    expanded: List[str] = []
    for b in base:
        expanded.append(b)
        if "%" in b:
            expanded.append(b.replace("%", "％"))
        if "％" in b:
            expanded.append(b.replace("％", "%"))
    monetary_units = ["원", "천원", "만원", "백만원", "억원"]
    if unit in monetary_units:
        for target_unit in monetary_units:
            if target_unit == unit:
                continue
            converted = _convert_unit(num_value, unit, target_unit)
            if converted is not None:
                if converted >= 1:
                    converted_int = int(converted)
                    converted_str = str(converted_int)
                    expanded.append(f"{converted_str}{target_unit}")
                    expanded.append(f"{_add_commas_to_int(converted_str)}{target_unit}")
                else:
                    expanded.append(f"{converted:.3f}{target_unit}".rstrip("0").rstrip("."))
    return unique_keep_order(expanded)


def extract_years(text: str) -> List[str]:
    return [m.group(0).replace(" ", "").replace("년", "") for m in _YEAR_PAT.finditer(text or "")]


def extract_units(text: str) -> List[str]:
    """Return which of UNITS appear in text."""
    t = normalize_invisibles(text)
    hits = [u for u in UNITS if u in t]
    return unique_keep_order(hits)


def extract_metrics(text: str) -> List[str]:
    t = (text or "").lower()
    found = []
    for canon, syns in METRIC_SYNONYMS.items():
        for s in syns:
            if s.lower() in t:
                found.append(canon)
                break
    return found


def has_metric(text: str) -> bool:
    return len(extract_metrics(text)) > 0


def _digits_only(n: str) -> str:
    return re.sub(r"\D", "", n or "")


def big_numbers_in_claim(nums: List[str]) -> List[str]:
    out = [n for n in nums if len(_digits_only(n)) >= 3]
    return unique_keep_order(out)


def match_numbers_in_text(nums: List[str], text_norm: str) -> Tuple[List[str], int]:
    matched = []
    exact_big = 0
    for n in nums:
        variants = number_variants(n)
        hit = next((v for v in variants if v and v in text_norm), None)
        if hit:
            matched.append(n)
            if len(_digits_only(n)) >= 3:
                exact_big += 1
    return unique_keep_order(matched), exact_big


# Unit factors: value in unit -> value in 백만원
_UNIT_TO_백만원 = {
    "원": 1 / 1_000_000,
    "천원": 1 / 1_000,
    "만원": 1 / 10,
    "백만원": 1.0,
    "억원": 100.0,
}


def parse_claim_number_to_백만원(token: str) -> Optional[float]:
    """Parse claim number string (e.g. '147,746백만원', '1억원') to value in 백만원."""
    token = normalize_invisibles(token).strip()
    if not token or not re.search(r"\d", token):
        return None
    m = re.match(r"^([-+]?)([0-9][0-9,]*(?:\.[0-9]+)?)(.*)$", token)
    if not m:
        return None
    sign, num_str, unit = m.group(1), m.group(2), m.group(3) or ""
    unit = normalize_invisibles(unit).strip()
    try:
        val = float(num_str.replace(",", ""))
    except ValueError:
        return None
    if sign == "-":
        val = -val
    factor = _UNIT_TO_백만원.get(unit, None)
    if factor is not None:
        return val * factor
    if not unit or unit in ("%", "％", "%p", "p", "배", "개월", "년"):
        return None
    return None


def match_numbers_against_xlsx_amounts(
    claim_nums: List[str],
    xlsx_amounts: List[Tuple[float, str, int, str, int]],
    tol: float = 0.01,
) -> Tuple[List[str], int]:
    """Match claim numbers against XLSX amounts (exact or 백만원-converted).
    xlsx_amounts: (value_백만원, cell_text, row, sheet, file_idx)
    Returns (matched_claim_nums, exact_big_match_count).
    """
    matched: List[str] = []
    exact_big = 0
    xlsx_vals = {a[0] for a in xlsx_amounts}
    for n in claim_nums:
        v_백만원 = parse_claim_number_to_백만원(n)
        if v_백만원 is None:
            continue
        hit = False
        for xv in xlsx_vals:
            if abs(xv - v_백만원) < tol or (xv != 0 and abs((xv - v_백만원) / xv) < tol):
                hit = True
                break
        if hit:
            matched.append(n)
            if len(_digits_only(n)) >= 3:
                exact_big += 1
    return unique_keep_order(matched), exact_big


def extract_word_tokens(text: str) -> List[str]:
    text = normalize_invisibles(text)
    return unique_keep_order(_WORD_RE.findall(text))
