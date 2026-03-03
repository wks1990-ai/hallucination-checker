"""Report helpers: HTML escaping, highlighting, CSV/Excel write."""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from verify_io import normalize_invisibles, unique_keep_order
from verify_numbers import extract_numbers, number_variants


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def highlight_snippet(snippet: str, claim: str) -> str:
    """Highlight claim words/numbers in evidence snippet with <b>."""
    if not snippet:
        return ""
    snippet_norm = normalize_invisibles(snippet)
    nums = extract_numbers(claim)
    num_tokens = []
    for n in nums:
        num_tokens.extend(number_variants(n))
    unit_words = {"원", "천원", "만원", "백만원", "억원", "%", "％", "배", "개월", "년", "%p", "p"}
    words = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}[A-Za-z0-9\-_/]*", normalize_invisibles(claim))
    words = [w for w in words if w not in unit_words]
    tokens = unique_keep_order(num_tokens + words)
    tokens = [t for t in tokens if t and t in snippet_norm]
    esc = html_escape(snippet_norm)
    for t in sorted(tokens, key=len, reverse=True):
        t_esc = html_escape(t)
        esc = re.sub(re.escape(t_esc), lambda m: f"<b>{m.group(0)}</b>", esc)
    return esc


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt


def safe_write_excel(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return None
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_excel(alt, index=False)
        return alt
