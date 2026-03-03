"""Chunking and search index building."""
from typing import List, Tuple

from verify_io import normalize_invisibles


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_search_index(pages: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Build chunks over PDF pages. meta: (page_number_1based, chunk_id)."""
    all_chunks: List[str] = []
    meta: List[Tuple[int, int]] = []
    for p_idx, p_text in enumerate(pages):
        chs = chunk_text(p_text)
        for c_idx, ch in enumerate(chs):
            all_chunks.append(ch)
            meta.append((p_idx + 1, c_idx))
    return all_chunks, meta


def build_search_index_from_rows(rows: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Build chunks over JSON rows. meta: (row_number_1based, chunk_id)."""
    all_chunks: List[str] = []
    meta: List[Tuple[int, int]] = []
    for r_idx, r_text in enumerate(rows):
        chs = chunk_text(r_text)
        for c_idx, ch in enumerate(chs):
            all_chunks.append(ch)
            meta.append((r_idx + 1, c_idx))
    return all_chunks, meta


def build_search_index_from_xlsx_rows(
    rows: List[str],
    row_meta: List[Tuple[int, int, str]],
) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """Build chunks over XLSX rows. row_meta[i] = (file_idx, row_1based, sheet_name).
    Returns (chunks, meta) where meta = (file_idx, row_1based, chunk_id).
    """
    all_chunks: List[str] = []
    meta: List[Tuple[int, int, int]] = []
    for r_idx, r_text in enumerate(rows):
        chs = chunk_text(r_text)
        file_idx, row_no, sheet = row_meta[r_idx] if r_idx < len(row_meta) else (0, r_idx + 1, "")
        for c_idx, ch in enumerate(chs):
            all_chunks.append(ch)
            meta.append((file_idx, row_no, c_idx))
    return all_chunks, meta
