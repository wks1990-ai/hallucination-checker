"""Claim splitting and table exclusion rules."""
import re
from typing import List

from verify_constants import EXCLUDED_TABLE_HEADERS, FIN_TABLE_KEYS
from verify_io import normalize_invisibles, protect_acronym_dots, restore_acronym_dots


def _normalize_heading_line(s: str) -> str:
    s = normalize_invisibles(s)
    s = s.strip()
    s = re.sub(r"^(?:[-*+]\s+)+", "", s)
    s = re.sub(r"^\\+(?=#)", "", s)
    return s.lstrip()


def _is_md_h2_or_more(s: str) -> bool:
    s2 = _normalize_heading_line(s)
    return bool(re.match(r"^#{1,6}\s*", s2))


def _is_numbered_title(s: str) -> bool:
    s = normalize_invisibles(s.strip())
    if re.match(r"^\d+\.\s+[\uAC00-\uD7A3A-Za-z\s]{1,20}$", s):
        return True
    if re.match(r"^\d+\.\s+[\uAC00-\uD7A3A-Za-z\s]{1,15}$", s):
        return True
    return False


def is_excluded_table_header(line: str) -> bool:
    if not line:
        return False
    line = normalize_invisibles(line)
    line = re.sub(r"^\s*(?:[-*+]\s+)+", "", line.strip())
    if not (line.startswith("|") and line.endswith("|")):
        return False
    cells = [c.strip() for c in line.strip("|").split("|")]
    cells = [c for c in cells if c]
    cell_set = set(cells)
    if EXCLUDED_TABLE_HEADERS.issubset(cell_set):
        return True
    hit = sum(1 for c in cells if c in FIN_TABLE_KEYS)
    return hit >= 4


def is_excluded_finance_data_row(line: str) -> bool:
    if not line:
        return False
    line = normalize_invisibles(line)
    line = re.sub(r"^\s*(?:[-*+]\s+)+", "", line.strip())
    if not (line.startswith("|") and line.endswith("|")):
        return False
    cells = [c.strip() for c in line.strip("|").split("|")]
    first = cells[0].strip() if cells else ""
    if not re.fullmatch(r"(?:19|20)\d{2}", first):
        return False
    num_pat = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|%p|p)?$")
    numeric = 0
    nonempty = 0
    has_percent = False
    for c in cells[1:]:
        c = c.strip()
        if not c:
            continue
        nonempty += 1
        c_norm = normalize_invisibles(c).replace(" ", "")
        if c_norm in {"-", "?", "?"}:
            continue
        if "%" in c_norm:
            has_percent = True
        c_num = re.sub(r"(�?씅|씅�|씅|씅|씅씅|씅)$", "", c_norm)
        if num_pat.match(c_num):
            numeric += 1
    return nonempty >= 6 and numeric >= 5 and (has_percent or numeric >= 7)


def split_claims(md_text: str) -> List[str]:
    """Split markdown into claim strings (code blocks/headers/tables handled)."""
    md_text = normalize_invisibles(md_text)
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"\[\^[^\]]+\]", "", md_text)

    kept_lines: List[str] = []
    for ln in md_text.splitlines():
        if _is_md_h2_or_more(ln):
            continue
        if _is_numbered_title(ln):
            continue
        if re.match(r"^\s*\(?\s*씅씅\s*[:씅]\s*[^)]+\)?\s*$", ln, re.IGNORECASE):
            continue
        ln_norm = normalize_invisibles(ln)
        if re.fullmatch(r"\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*", ln_norm):
            continue
        if is_excluded_table_header(ln) or is_excluded_finance_data_row(ln):
            continue
        kept_lines.append(ln)
    md_text = "\n".join(kept_lines)

    md_text = md_text.replace("\u00A0", " ")
    md_text = md_text.translate(str.maketrans({"씅": "(", "씅": ")", "씅": "+", "씅": "-"}))
    md_text = re.sub(r"\s*\(\s*[+-]\s*\)\s*", "\n\n", md_text)
    md_text = re.sub(r"(?m)(^|\s)\+\s+(?=\D)", r"\1\n\n", md_text)
    md_text = re.sub(r"(?m)(^|\s)-\s+(?=\D)", r"\1\n\n", md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    md_text = protect_acronym_dots(md_text)

    blocks = [b.strip() for b in re.split(r"\n{2,}", md_text) if b.strip()]
    claims: List[str] = []
    dot_boundary = r"(?<!\d)\.(?!\d)"
    ender_pattern = rf"(씅\.|씅\.|씅\.|씅\.|�?�\.|{dot_boundary})"

    for b in blocks:
        if is_excluded_table_header(b.strip()) or is_excluded_finance_data_row(b.strip()):
            continue
        if "|" in b and "\n" in b and re.search(r"^\|.*\|$", b.strip(), flags=re.MULTILINE):
            sep_pat = r"\|\s*[-: ]+\s*(\|\s*[-: ]+\s*)+\|"
            header_line = None
            for _ln in b.splitlines():
                _ln = normalize_invisibles(_ln.strip())
                if not _ln:
                    continue
                if re.fullmatch(sep_pat, _ln):
                    continue
                if _ln.startswith("|") and _ln.endswith("|"):
                    header_line = _ln
                    break
            if header_line and is_excluded_table_header(header_line):
                continue
            for line in b.splitlines():
                line = normalize_invisibles(line.strip())
                if not line:
                    continue
                if _is_md_h2_or_more(line):
                    continue
                if re.fullmatch(sep_pat, line):
                    continue
                if line.startswith("|") and line.endswith("|"):
                    line = re.sub(r"\s+", " ", line).strip()
                    if is_excluded_table_header(line) or is_excluded_finance_data_row(line):
                        continue
                    if len(line) >= 8:
                        claims.append(line)
            continue

        parts = re.split(ender_pattern, b)
        buf = ""
        for p in parts:
            if not p:
                continue
            buf += p
            if re.fullmatch(ender_pattern, p):
                claim = re.sub(r"\s+", " ", normalize_invisibles(buf)).strip()
                claim = re.sub(r"\s(?:-{3,}|\*{3,}|_{3,})\s", " ", claim)
                if _is_md_h2_or_more(claim):
                    buf = ""
                    continue
                if len(claim) >= 8:
                    claims.append(claim)
                buf = ""
        tail = re.sub(r"\s+", " ", normalize_invisibles(buf)).strip()
        tail = re.sub(r"\s(?:-{3,}|\*{3,}|_{3,})\s", " ", tail)
        if tail and (not _is_md_h2_or_more(tail)) and len(tail) >= 8:
            claims.append(tail)

    claims = [c for c in claims if not _is_md_h2_or_more(c)]
    claims = [c for c in claims if not re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", c)]
    claims = [c for c in claims if not _is_numbered_title(c)]
    claims = [c for c in claims if not re.match(r"^\s*\(?\s*씅씅\s*[:씅]\s*[^)]+\)?\s*$", c, re.IGNORECASE)]
    claims = [restore_acronym_dots(c) for c in claims]
    return claims
