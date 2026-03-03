"""IO, text normalization, PDF/JSON/XLSX extraction."""
import json as _json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


def normalize_invisibles(s: str) -> str:
    """Normalize invisible/special spaces and punctuation variants."""
    if not s:
        return s
    s = (s.replace("\u00A0", " ")
         .replace("\u202F", " ")
         .replace("\u2009", " ")
         .replace("\u200A", " ")
         .replace("\u2007", " "))
    s = s.replace("\uFEFF", "")
    s = re.sub(r"[\u200B\u200C\u200D\u2060]", "", s)
    s = (s.replace("\uFF03", "#")
         .replace("\uFF05", "%")
         .replace("\uFF0C", ",")
         .replace("\uFF0E", "."))
    s = s.replace("\uFFFD", " ")
    s = re.sub(r"(?<=[0-9A-Za-z\uAC00-\uD7A3])\?(?=[0-9A-Za-z\uAC00-\uD7A3])", " ", s)
    return s


_DOT_TOKEN = "?DOT?"
_ACRONYM_DOT_RE = re.compile(r"(?i)(?:[A-Z]\.){2,}[A-Z]?")


def protect_acronym_dots(s: str) -> str:
    """Replace '.' inside acronym-like patterns with a token."""
    if not s:
        return s
    return _ACRONYM_DOT_RE.sub(lambda m: m.group(0).replace(".", _DOT_TOKEN), s)


def restore_acronym_dots(s: str) -> str:
    """Restore token back to '.'."""
    if not s:
        return s
    return s.replace(_DOT_TOKEN, ".")


def read_text_file(path: Path) -> str:
    """Read .md/.txt robustly across common Korean encodings."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return normalize_invisibles(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return normalize_invisibles(raw.decode("utf-8", errors="replace"))


def extract_pdf_text_by_page(pdf_path: Path) -> List[str]:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages: List[str] = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        text = normalize_invisibles(text)
        text = re.sub(r"[ \t]+", " ", text)
        pages.append(text.strip())
    return pages


def _flatten_to_strings(obj) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        s = normalize_invisibles(obj).strip()
        if s:
            out.append(s)
        return out
    if isinstance(obj, (int, float, bool)):
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_to_strings(v))
        return out
    if isinstance(obj, list):
        for it in obj:
            out.extend(_flatten_to_strings(it))
        return out
    return out


def extract_json_text_rows(json_path: Path) -> List[str]:
    """Load JSON and return list of row texts (supports list[str], list[dict], nested)."""
    raw = json_path.read_text(encoding="utf-8", errors="ignore")
    data = _json.loads(raw)
    if isinstance(data, dict):
        for k in ("rows", "data", "items", "documents", "docs"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    rows: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                txt = normalize_invisibles(item).strip()
                if txt:
                    rows.append(txt)
            elif isinstance(item, dict):
                parts = _flatten_to_strings(item)
                txt = " ".join(parts).strip()
                if txt:
                    rows.append(txt)
            else:
                parts = _flatten_to_strings(item)
                txt = " ".join(parts).strip()
                if txt:
                    rows.append(txt)
    elif isinstance(data, dict):
        parts = _flatten_to_strings(data)
        txt = " ".join(parts).strip()
        if txt:
            rows.append(txt)
    else:
        parts = _flatten_to_strings(data)
        txt = " ".join(parts).strip()
        if txt:
            rows.append(txt)
    rows = [re.sub(r"[ \t]+", " ", normalize_invisibles(r)).strip() for r in rows if str(r).strip()]
    return rows


def extract_xlsx_rows(xlsx_path: Path) -> List[Tuple[str, int, str]]:
    """Load XLSX and return list of (row_text, row_index_1based, sheet_name).
    All sheets are flattened; row_text = joined cell values as string for similarity search.
    """
    try:
        dfs = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    except Exception:
        return []
    rows: List[Tuple[str, int, str]] = []
    for sheet_name, df in dfs.items():
        if df is None or df.empty:
            continue
        for idx, row in df.iterrows():
            parts = []
            for v in row:
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    parts.append(normalize_invisibles(s))
            txt = " ".join(parts).strip()
            if txt:
                rows.append((txt, int(idx) + 1, str(sheet_name)))
    return rows


def extract_xlsx_amounts_백만원(xlsx_path: Path) -> List[Tuple[float, str, int, str]]:
    """Extract all monetary amounts from XLSX, normalized to 백만원.
    Returns list of (value_백만원, cell_text, row_1based, sheet_name).
    Detects unit from column header (백만원, 억원, 만원, 천원, 원) or assumes 백만원 for raw numbers.
    """
    unit_to_factor = {
        "원": 1 / 1_000_000,
        "천원": 1 / 1_000,
        "만원": 1 / 10,
        "백만원": 1.0,
        "억원": 100.0,
    }
    try:
        dfs = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    except Exception:
        return []
    amounts: List[Tuple[float, str, int, str]] = []
    num_pat = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")

    for sheet_name, df in dfs.items():
        if df is None or df.empty:
            continue
        # Infer unit from first row (headers)
        header_row = df.iloc[0] if len(df) > 0 else pd.Series()
        col_units: dict = {}
        for cidx, h in enumerate(header_row):
            hstr = str(h).strip() if pd.notna(h) else ""
            hstr = normalize_invisibles(hstr)
            if "%" in hstr or "퍼센트" in hstr:
                col_units[cidx] = None  # skip percentage columns
                continue
            for u, _ in unit_to_factor.items():
                if u in hstr:
                    col_units[cidx] = u
                    break
            if cidx not in col_units:
                col_units[cidx] = "백만원"

        for row_idx, row in df.iterrows():
            for cidx, v in enumerate(row):
                if pd.isna(v):
                    continue
                s = str(v).strip()
                s_nocomma = s.replace(",", "")
                if not re.search(r"\d", s):
                    continue
                unit = col_units.get(cidx)
                if unit is None:
                    continue
                try:
                    val = float(s_nocomma)
                except ValueError:
                    # e.g. "147,746백만원"
                    m = re.match(r"^(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*(백만원|억원|만원|천원|원)?$", s)
                    if m:
                        val = float(m.group(1).replace(",", ""))
                        u2 = m.group(2) or unit
                        factor = unit_to_factor.get(u2, 1.0)
                    else:
                        continue
                else:
                    factor = unit_to_factor.get(unit, 1.0)
                value_백만원 = val * factor
                amounts.append((value_백만원, s, int(row_idx) + 1, str(sheet_name)))
    return amounts


def extract_xlsx_text_rows_from_paths(
    xlsx_paths: List[Path],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Load multiple XLSX files and return (all_row_texts, row_meta).
    row_meta[i] = (file_idx, row_1based, sheet_name) for row i.
    """
    all_texts: List[str] = []
    row_meta: List[Tuple[int, int, str]] = []
    for fidx, p in enumerate(xlsx_paths):
        for txt, row_no, sheet in extract_xlsx_rows(p):
            all_texts.append(txt)
            row_meta.append((fidx, row_no, sheet))
    return all_texts, row_meta


def extract_xlsx_amounts_from_paths(xlsx_paths: List[Path]) -> List[Tuple[float, str, int, str, int]]:
    """Load amounts from all XLSX files. Returns (value_백만원, cell_text, row_1based, sheet_name, file_idx)."""
    result: List[Tuple[float, str, int, str, int]] = []
    for fidx, p in enumerate(xlsx_paths):
        for val, cell_txt, row_no, sheet in extract_xlsx_amounts_백만원(p):
            result.append((val, cell_txt, row_no, sheet, fidx))
    return result


def unique_keep_order(items: List[str]) -> List[str]:
    """Deduplicate list while preserving order."""
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def norm_for_match(s: str) -> str:
    """Light normalization for matching (lower English; collapse spaces)."""
    s = normalize_invisibles(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s
