"""Scoring: coverage, proximity, direction, fact_block, status."""
import re
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from verify_constants import DIR_NEG, DIR_OPPOSITES, DIR_POS, INDICATORS
from verify_io import normalize_invisibles, norm_for_match
from verify_numbers import (
    _digits_only,
    extract_metrics,
    extract_units,
    extract_word_tokens,
    extract_years,
    has_metric,
    match_numbers_in_text,
    number_variants,
    numbers_in_claim,
)


def _split_sentences_kor(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n\.\?\!]+", text)
    return [p.strip() for p in parts if p.strip()]


def fact_block_score(claim_text: str, evidence_text: str, window: int = 3) -> float:
    """Return 0~1.0 score based on co-occurrence of (metric, number, unit, year) within a local window."""
    claim = claim_text or ""
    ev = evidence_text or ""
    c_metrics = extract_metrics(claim)
    c_years = extract_years(claim)
    c_units = extract_units(claim)
    c_nums, _ = match_numbers_in_text(numbers_in_claim(claim), norm_for_match(claim)) if claim else ([], 0)
    if not c_metrics or not c_nums:
        return 0.0
    sents = _split_sentences_kor(ev)
    if not sents:
        return 0.0
    best = 0.0
    for i in range(len(sents)):
        win = " ".join(sents[i : i + window])
        w_metrics = extract_metrics(win)
        w_years = extract_years(win)
        w_units = extract_units(win)
        w_nums, _ = match_numbers_in_text(numbers_in_claim(win), norm_for_match(win))
        metric_ok = any(m in w_metrics for m in c_metrics)
        num_ok = any(n in w_nums for n in c_nums)
        year_ok = (not c_years) or any(y in w_years for y in c_years)
        unit_ok = (not c_units) or any(u in w_units for u in c_units)
        if (not metric_ok) and (not num_ok):
            continue
        if metric_ok and num_ok and (year_ok or unit_ok) and (year_ok and unit_ok):
            score = 1.0
        elif metric_ok and num_ok and (year_ok or unit_ok):
            score = 0.85
        elif metric_ok and num_ok:
            score = 0.65
        elif metric_ok and (year_ok or unit_ok):
            score = 0.45
        else:
            score = 0.20
        best = max(best, score)
    return best


def build_word_idf_map(texts: List[str]) -> Dict[str, float]:
    token_pattern = r"(?u)(?:[가-힣]{2,}|[A-Za-z]{2,}[A-Za-z0-9\-_/]*|\d+(?:\.\d+)?)"
    vec = TfidfVectorizer(analyzer="word", token_pattern=token_pattern, lowercase=True, min_df=1)
    try:
        X = vec.fit_transform([normalize_invisibles(t) for t in texts if t and t.strip()])
    except ValueError:
        return {}
    if not hasattr(vec, "vocabulary_") or not hasattr(vec, "idf_"):
        return {}
    idf = vec.idf_
    vocab = vec.vocabulary_
    return {tok: float(idf[idx]) for tok, idx in vocab.items()}


def token_coverage_idf_ratio(
    claim: str,
    source_text: str,
    nums: List[str],
    idf_map: Dict[str, float],
) -> float:
    from verify_io import unique_keep_order

    claim_n = normalize_invisibles(claim)
    src_norm = norm_for_match(source_text)
    words = extract_word_tokens(claim_n)
    stop = {
        "그리고", "또한", "하지만", "따라서", "즉", "등", "및",
        "으로", "에서", "대비", "전년", "금년", "당기", "동기",
        "수준", "기준", "확인", "예정", "관련", "등의",
    }
    words = [w for w in words if w and w not in stop]
    num_tokens = nums[:]
    tokens = unique_keep_order(words + num_tokens)
    if not tokens:
        return 0.0

    def _w_word(t: str) -> float:
        key = norm_for_match(t).strip()
        return float(idf_map.get(key, 1.0)) * max(1, len(t))

    def _w_num(t: str) -> float:
        d = _digits_only(t)
        base = max(1, len(d) if d else len(t))
        boost = 2.0 if len(d) >= 3 else 1.0
        return boost * base

    matched_w = 0.0
    total_w = 0.0
    for t in sorted(tokens, key=len, reverse=True):
        if not t:
            continue
        w = _w_num(t) if t in num_tokens else _w_word(t)
        total_w += w
        if t in num_tokens:
            if any(v.lower() in src_norm for v in number_variants(t)):
                matched_w += w
        else:
            if norm_for_match(t) in src_norm:
                matched_w += w
    if total_w <= 0:
        return 0.0
    return max(0.0, min(1.0, matched_w / total_w))


def token_coverage_blend(
    claim: str,
    evidence_union: str,
    source_full: str,
    nums: List[str],
    idf_map: Dict[str, float],
) -> float:
    cov_local = token_coverage_idf_ratio(claim, evidence_union, nums, idf_map) if evidence_union else 0.0
    cov_global = token_coverage_idf_ratio(claim, source_full, nums, idf_map) if source_full else 0.0
    blended = max(cov_local, 0.85 * cov_global + 0.15 * cov_local)
    return max(0.0, min(1.0, blended))


def unit_match_ratio(claim: str, evidence_union: str) -> float:
    claim_units = extract_units(claim)
    if not claim_units:
        return 0.5
    src = normalize_invisibles(evidence_union)
    hit = sum(1 for u in claim_units if u in src)
    return hit / max(1, len(claim_units))


def direction_consistency(claim: str, evidence_union: str) -> Tuple[float, bool]:
    c = normalize_invisibles(claim)
    e = normalize_invisibles(evidence_union)
    claim_dirs = [w for w in (DIR_POS | DIR_NEG) if w in c]
    if not claim_dirs:
        return 0.5, False
    conflict = False
    for a, b in DIR_OPPOSITES:
        if a in c and b in e:
            conflict = True
            break
        if b in c and a in e:
            conflict = True
            break
    hit = sum(1 for w in claim_dirs if w in e)
    ratio = hit / max(1, len(claim_dirs))
    if conflict:
        return 0.0, True
    return 0.5 + 0.5 * ratio, False


def indicator_number_proximity_ratio(
    claim: str, evidence_union: str, nums: List[str], window: int = 40
) -> float:
    from verify_io import unique_keep_order

    c = normalize_invisibles(claim)
    e_norm = normalize_invisibles(evidence_union)
    if not e_norm:
        return 0.0 if nums else 0.5
    num_pos = []
    for n in nums:
        n_raw = normalize_invisibles(n).replace(" ", "")
        idx = c.replace(" ", "").find(n_raw)
        if idx != -1:
            num_pos.append((n, idx))
    pairs = []
    for ind in INDICATORS:
        p = c.find(ind)
        if p == -1:
            continue
        nearest = None
        best_d = 10**9
        for n, np in num_pos:
            d = abs(np - p)
            if d < best_d:
                best_d = d
                nearest = n
        if nearest is not None and best_d <= 80:
            pairs.append((ind, nearest))
    pairs = unique_keep_order([f"{a}|||{b}" for a, b in pairs])
    pairs = [(p.split("|||")[0], p.split("|||")[1]) for p in pairs]
    if not pairs:
        return 0.5

    def _has_near(ind: str, num: str) -> bool:
        start = 0
        while True:
            ip = e_norm.find(ind, start)
            if ip == -1:
                break
            lo = max(0, ip - window)
            hi = min(len(e_norm), ip + len(ind) + window)
            seg = e_norm[lo:hi]
            for v in number_variants(num):
                if v and v in seg:
                    return True
            start = ip + len(ind)
        return False

    hit = sum(1 for ind, num in pairs if _has_near(ind, num))
    return hit / max(1, len(pairs))


def combine_final_score(
    sim_best: float,
    coverage: float,
    proximity: float,
    big_num_recall: float,
    unit_ratio: float,
    direction_score: float,
    direction_conflict: bool,
    has_numbers: bool,
    fact_block: float = 0.0,
) -> float:
    if not has_numbers:
        w_sim, w_cov, w_fb, w_prox, w_big, w_unit, w_dir = 0.55, 0.40, 0.00, 0.03, 0.00, 0.01, 0.01
    else:
        w_sim, w_cov, w_fb, w_prox, w_big, w_unit, w_dir = 0.32, 0.22, 0.22, 0.12, 0.08, 0.03, 0.01
    score = (
        w_sim * sim_best + w_cov * coverage + w_fb * fact_block + w_prox * proximity
        + w_big * big_num_recall + w_unit * unit_ratio + w_dir * direction_score
    )
    if direction_conflict:
        score = max(0.0, score - 0.15)
    return max(0.0, min(1.0, score))


def decide_status(
    final_score: float,
    sim_best: float,
    coverage: float,
    big_num_recall: float,
    exact_big_matches: int,
    has_numbers: bool,
    direction_conflict: bool,
    fact_block: float = 0.0,
    has_metric: bool = False,
    th_num_supported: float = 0.30,
    th_num_partial: float = 0.22,
    th_text_supported: float = 0.30,
    th_text_partial: float = 0.22,
) -> str:
    if direction_conflict:
        if has_numbers:
            return "PARTIAL" if final_score >= th_num_partial else "UNSUPPORTED"
        return "PARTIAL" if final_score >= th_text_partial else "UNSUPPORTED"
    if has_numbers:
        if final_score >= th_num_supported and (
            fact_block >= 0.65
            or (has_metric and big_num_recall > 0.0 and coverage >= 0.45)
            or exact_big_matches >= 1
        ):
            return "SUPPORTED"
        if final_score >= th_num_partial and (
            fact_block >= 0.35 or big_num_recall > 0.0 or coverage >= 0.30 or sim_best >= 0.20
        ):
            return "PARTIAL"
        if fact_block >= 0.55 and (coverage >= 0.35 or sim_best >= 0.25):
            return "PARTIAL"
        return "UNSUPPORTED"
    if final_score >= th_text_supported and coverage >= 0.45:
        return "SUPPORTED"
    if final_score >= th_text_partial and coverage >= 0.30:
        return "PARTIAL"
    if coverage >= 0.55 and sim_best >= 0.18:
        return "PARTIAL"
    return "UNSUPPORTED"


def combine_status(status_pdf: str, status_json: str) -> str:
    if status_pdf == "SUPPORTED" or status_json == "SUPPORTED":
        return "SUPPORTED"
    if status_pdf == "PARTIAL" or status_json == "PARTIAL":
        return "PARTIAL"
    return "UNSUPPORTED"
