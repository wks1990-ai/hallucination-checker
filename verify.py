import argparse
import re
import json
import warnings

from verify_complete_gui import run_gui
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from verify_constants import STATUS_KO
from verify_models import Evidence, ClaimResult
from verify_io import (
    read_text_file,
    normalize_invisibles,
    extract_pdf_text_by_page,
    extract_xlsx_text_rows_from_paths,
    extract_xlsx_amounts_from_paths,
    unique_keep_order,
    norm_for_match,
)
from verify_chunking import build_search_index, build_search_index_from_xlsx_rows
from verify_claims import split_claims
from verify_numbers import (
    extract_numbers,
    numbers_in_claim,
    big_numbers_in_claim,
    match_numbers_in_text,
    match_numbers_against_xlsx_amounts,
    has_metric,
)
from verify_scoring import (
    build_word_idf_map,
    token_coverage_blend,
    unit_match_ratio,
    direction_consistency,
    indicator_number_proximity_ratio,
    combine_final_score,
    decide_status,
    combine_status,
    fact_block_score,
)
from verify_report import html_escape, highlight_snippet, safe_write_csv, safe_write_excel


# -----------------------------
# Main verify
# -----------------------------
def verify(
    md_path: Path,
    pdf_path: Path,
    out_dir: Path,
    top_k: int = 3,
    xlsx_paths: List[Path] | None = None,
    th_num_supported: float = 0.30,
    th_num_partial: float = 0.22,
    th_text_supported: float = 0.30,
    th_text_partial: float = 0.22,
) -> Dict:
    md_text = read_text_file(md_path)
    claims = split_claims(md_text)

    pages = extract_pdf_text_by_page(pdf_path)
    all_chunks, meta = build_search_index(pages)

    if not all_chunks:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다. (스캔 이미지 PDF일 가능성)")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
    X = vectorizer.fit_transform(all_chunks)
    idf_pdf = build_word_idf_map(all_chunks)

    results: List[ClaimResult] = []
    pdf_joined = normalize_invisibles("\n".join(pages))

    # XLSX sources (optional, 5~6 files)
    xlsx_paths = xlsx_paths or []
    all_chunks_xlsx: List[str] = []
    meta_xlsx: List[Tuple[int, int, int]] = []  # (file_idx, row_no, chunk_id)
    row_meta_xlsx: List[Tuple[int, int, str]] = []  # (file_idx, row_no, sheet)
    vectorizer_xlsx = None
    X_xlsx = None
    xlsx_joined = ""
    idf_xlsx: Dict[str, float] = {}
    xlsx_amounts: List[Tuple[float, str, int, str, int]] = []
    xlsx_paths_loaded: List[Path] = []
    if xlsx_paths:
        xlsx_paths_loaded = [p for p in xlsx_paths if p.exists()]
        if not xlsx_paths_loaded:
            raise FileNotFoundError(f"XLSX file(s) not found: {xlsx_paths}")
        xlsx_rows, row_meta_xlsx = extract_xlsx_text_rows_from_paths(xlsx_paths_loaded)
        xlsx_amounts = extract_xlsx_amounts_from_paths(xlsx_paths_loaded)
        if xlsx_rows:
            all_chunks_xlsx, meta_xlsx = build_search_index_from_xlsx_rows(xlsx_rows, row_meta_xlsx)
            if all_chunks_xlsx:
                vectorizer_xlsx = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
                X_xlsx = vectorizer_xlsx.fit_transform(all_chunks_xlsx)
                idf_xlsx = build_word_idf_map(all_chunks_xlsx)
                xlsx_joined = normalize_invisibles("\n".join(xlsx_rows))

    for i, claim in enumerate(claims, start=1):
        claim_norm = normalize_invisibles(claim)

        nums = extract_numbers(claim_norm)
        has_numbers = bool(nums)

        # ---- similarity + TOP-K evidence (PDF) ----
        q = vectorizer.transform([claim_norm])
        sims = cosine_similarity(q, X).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        sim_best_pdf = float(sims[top_idx[0]]) if len(top_idx) else 0.0

        ev_list: List[Evidence] = []
        pdf_union_chunks: List[str] = []
        for idx in top_idx:
            score = float(sims[idx])
            page, chunk_id = meta[idx]
            full_chunk = all_chunks[idx]
            pdf_union_chunks.append(full_chunk)
            snippet = full_chunk[:500].strip().replace("\n", " ")
            ev_list.append(Evidence(page=page, chunk_id=chunk_id, score=score, snippet=snippet))

        pdf_union_text = normalize_invisibles("\n".join(pdf_union_chunks))

        # ---- similarity + TOP-K evidence (XLSX) ----
        sim_best_xlsx = 0.0
        ev_list_xlsx: List[Evidence] = []
        xlsx_union_text = ""
        if vectorizer_xlsx is not None and X_xlsx is not None and all_chunks_xlsx:
            qx = vectorizer_xlsx.transform([claim_norm])
            sims_x = cosine_similarity(qx, X_xlsx).flatten()
            top_idx_x = sims_x.argsort()[::-1][:top_k]
            sim_best_xlsx = float(sims_x[top_idx_x[0]]) if len(top_idx_x) else 0.0

            xlsx_union_chunks: List[str] = []
            for jdx in top_idx_x:
                jscore = float(sims_x[jdx])
                file_idx, row_no, chunk_id = meta_xlsx[jdx]
                full_chunk = all_chunks_xlsx[jdx]
                xlsx_union_chunks.append(full_chunk)
                fname = xlsx_paths_loaded[file_idx].name if file_idx < len(xlsx_paths_loaded) else ""
                jsnippet = f"[{fname}] row {row_no}: " + full_chunk[:450].strip().replace("\n", " ")
                ev_list_xlsx.append(Evidence(page=row_no, chunk_id=chunk_id, score=jscore, snippet=jsnippet))

            xlsx_union_text = normalize_invisibles("\n".join(xlsx_union_chunks))

        # ---- number presence: PDF text + XLSX amounts(백만원) + XLSX text ----
        matched_pdf_any, _ = match_numbers_in_text(nums, pdf_joined)
        matched_xlsx_amt, _ = match_numbers_against_xlsx_amounts(nums, xlsx_amounts) if xlsx_amounts else ([], 0)
        matched_xlsx_text, _ = match_numbers_in_text(nums, xlsx_joined) if xlsx_joined else ([], 0)
        matched_xlsx_any = unique_keep_order(matched_xlsx_amt + matched_xlsx_text)

        matched_union_any = unique_keep_order(matched_pdf_any + matched_xlsx_any)

        missing_pdf = [n for n in nums if n not in matched_pdf_any]
        missing_xlsx = [n for n in nums if n not in matched_xlsx_any] if xlsx_joined else nums[:]
        missing_all = [n for n in nums if n not in matched_union_any]

        # ---- richer scoring on TOP-K union (more faithful evidence) ----
        # PDF features
        cov_pdf = token_coverage_blend(claim_norm, pdf_union_text, pdf_joined, nums, idf_pdf)
        prox_pdf = indicator_number_proximity_ratio(claim_norm, pdf_union_text, nums)
        unit_pdf = unit_match_ratio(claim_norm, pdf_union_text)
        dir_pdf, dir_conflict_pdf = direction_consistency(claim_norm, pdf_union_text)
        fb_pdf = fact_block_score(claim, pdf_union_text)
        hm_claim = has_metric(claim)


        big_nums = big_numbers_in_claim(nums)
        if big_nums:
            matched_big_pdf, _ = match_numbers_in_text(big_nums, norm_for_match(pdf_union_text))
            exact_big_pdf = len(matched_big_pdf)
            big_recall_pdf = exact_big_pdf / max(1, len(big_nums))
        elif nums:
            matched_evid_pdf, _ = match_numbers_in_text(nums, norm_for_match(pdf_union_text))
            exact_big_pdf = 0
            big_recall_pdf = len(matched_evid_pdf) / max(1, len(nums))
        else:
            exact_big_pdf = 0
            big_recall_pdf = 0.5

        final_pdf = combine_final_score(
            sim_best=sim_best_pdf,
            coverage=cov_pdf,
            proximity=prox_pdf,
            big_num_recall=big_recall_pdf,
            unit_ratio=unit_pdf,
            direction_score=dir_pdf,
            direction_conflict=dir_conflict_pdf,
            has_numbers=has_numbers,
            fact_block=fb_pdf,
        )

        # XLSX features (금액 백만원 일치 중요)
        cov_xlsx = 0.0
        prox_xlsx = 0.0
        unit_xlsx = 0.5
        dir_xlsx = 0.5
        dir_conflict_xlsx = False
        big_recall_xlsx = 0.5 if has_numbers else 0.5
        exact_big_xlsx = 0
        final_xlsx = 0.0
        fb_xlsx = 0.0

        if xlsx_union_text:
            cov_xlsx = token_coverage_blend(claim_norm, xlsx_union_text, xlsx_joined, nums, idf_xlsx)
            prox_xlsx = indicator_number_proximity_ratio(claim_norm, xlsx_union_text, nums)
            unit_xlsx = unit_match_ratio(claim_norm, xlsx_union_text)
            dir_xlsx, dir_conflict_xlsx = direction_consistency(claim_norm, xlsx_union_text)
            fb_xlsx = fact_block_score(claim, xlsx_union_text)

            # 금액: 백만원 환산 일치 우선 반영
            matched_xlsx_big, exact_big_xlsx = match_numbers_against_xlsx_amounts(big_nums or nums, xlsx_amounts)
            big_recall_xlsx = len(matched_xlsx_big) / max(1, len(big_nums or nums)) if (big_nums or nums) else 0.5

            final_xlsx = combine_final_score(
                sim_best=sim_best_xlsx,
                coverage=cov_xlsx,
                proximity=prox_xlsx,
                big_num_recall=big_recall_xlsx,
                unit_ratio=unit_xlsx,
                direction_score=dir_xlsx,
                direction_conflict=dir_conflict_xlsx,
                has_numbers=has_numbers,
                fact_block=fb_xlsx,
            )

        # overall
        best_pdf = final_pdf
        best_xlsx = final_xlsx
        best_overall = max(best_pdf, best_xlsx)

        status_pdf = decide_status(
            best_pdf, sim_best_pdf, cov_pdf, big_recall_pdf, exact_big_pdf, has_numbers, dir_conflict_pdf, fb_pdf, hm_claim,
            th_num_supported=th_num_supported, th_num_partial=th_num_partial,
            th_text_supported=th_text_supported, th_text_partial=th_text_partial,
        )

        status_xlsx = decide_status(
            best_xlsx, sim_best_xlsx, cov_xlsx, big_recall_xlsx, exact_big_xlsx, has_numbers, dir_conflict_xlsx, fb_xlsx, hm_claim,
            th_num_supported=th_num_supported, th_num_partial=th_num_partial,
            th_text_supported=th_text_supported, th_text_partial=th_text_partial,
        ) if xlsx_joined else "UNSUPPORTED"

        status_overall = combine_status(status_pdf, status_xlsx)

        # 숫자 이슈(불일치/누락) 의심 플래그
        num_conflict_pdf = bool(nums) and bool(ev_list) and (best_pdf >= th_num_partial) and (len(missing_pdf) > 0)
        num_conflict_xlsx = bool(nums) and bool(ev_list_xlsx) and (best_xlsx >= th_num_partial) and (len(missing_xlsx) > 0)

        results.append(
            ClaimResult(
                claim_id=i,
                claim_text=claim_norm,

                status_overall=status_overall,
                status_pdf=status_pdf,
                status_xlsx=status_xlsx,

                best_score=best_overall,
                best_score_pdf=best_pdf,
                best_score_xlsx=best_xlsx,

                sim_best_pdf=sim_best_pdf,
                sim_best_xlsx=sim_best_xlsx,

                coverage_pdf=cov_pdf,
                coverage_xlsx=cov_xlsx,

                proximity_pdf=prox_pdf,
                proximity_xlsx=prox_xlsx,

                big_num_recall_pdf=big_recall_pdf,
                big_num_recall_xlsx=big_recall_xlsx,

                exact_big_num_matches_pdf=exact_big_pdf,
                exact_big_num_matches_xlsx=exact_big_xlsx,

                unit_match_pdf=unit_pdf,
                unit_match_xlsx=unit_xlsx,

                direction_score_pdf=dir_pdf,
                direction_score_xlsx=dir_xlsx,

                direction_conflict_pdf=dir_conflict_pdf,
                direction_conflict_xlsx=dir_conflict_xlsx,

                numbers_in_claim=nums,
                matched_numbers=matched_union_any,
                matched_numbers_pdf=matched_pdf_any,
                matched_numbers_xlsx=matched_xlsx_any,

                missing_numbers_pdf=missing_pdf,
                missing_numbers_xlsx=missing_xlsx,
                missing_numbers_all=missing_all,

                num_conflict_pdf=num_conflict_pdf,
                num_conflict_xlsx=num_conflict_xlsx,

                evidence=ev_list,
                evidence_xlsx=ev_list_xlsx,
            )
        )


    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_json_path = out_dir / "report.json"
    report_data = {
        "md_file": str(md_path),
        "pdf_file": str(pdf_path),
        "xlsx_source_files": [str(p) for p in xlsx_paths_loaded],
        "report_json": str(report_json_path),
        "total_claims": len(results),
        "thresholds": {
            "th_num_supported": th_num_supported,
            "th_num_partial": th_num_partial,
            "th_text_supported": th_text_supported,
            "th_text_partial": th_text_partial,
        },
        "results": [asdict(r) for r in results],
    }
    report_json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV / XLSX
    rows = []
    for r in results:
        e0 = r.evidence[0] if r.evidence else None
        x0 = r.evidence_xlsx[0] if r.evidence_xlsx else None

        num_total = len(r.numbers_in_claim)
        pdf_num_matched = len(r.matched_numbers_pdf)
        xlsx_num_matched = len(r.matched_numbers_xlsx)
        num_missing_all = len(r.missing_numbers_all)

        rows.append(
            {
                "claim_id": r.claim_id,

                # statuses
                "status_overall": STATUS_KO.get(r.status_overall, r.status_overall),
                "raw_status_overall": r.status_overall,
                "status_pdf": STATUS_KO.get(r.status_pdf, r.status_pdf),
                "raw_status_pdf": r.status_pdf,
                "status_xlsx": STATUS_KO.get(r.status_xlsx, r.status_xlsx),
                "raw_status_xlsx": r.status_xlsx,

                # scores (final)
                "best_score_overall": r.best_score,
                "best_score_pdf": r.best_score_pdf,
                "best_score_xlsx": r.best_score_xlsx,

                # raw similarity (TF-IDF cosine)
                "sim_best_pdf": r.sim_best_pdf,
                "sim_best_xlsx": r.sim_best_xlsx,

                # feature scores (0~1)
                "coverage_pdf": r.coverage_pdf,
                "coverage_xlsx": r.coverage_xlsx,
                "proximity_pdf": r.proximity_pdf,
                "proximity_xlsx": r.proximity_xlsx,
                "big_num_recall_pdf": r.big_num_recall_pdf,
                "big_num_recall_xlsx": r.big_num_recall_xlsx,
                "exact_big_num_matches_pdf": r.exact_big_num_matches_pdf,
                "exact_big_num_matches_xlsx": r.exact_big_num_matches_xlsx,
                "unit_match_pdf": r.unit_match_pdf,
                "unit_match_xlsx": r.unit_match_xlsx,
                "direction_score_pdf": r.direction_score_pdf,
                "direction_score_xlsx": r.direction_score_xlsx,
                "direction_conflict_pdf": int(bool(r.direction_conflict_pdf)),
                "direction_conflict_xlsx": int(bool(r.direction_conflict_xlsx)),


                # numbers (counts + lists)
                "num_total": num_total,
                "pdf_num_matched": pdf_num_matched,
                "xlsx_num_matched": xlsx_num_matched,
                "num_missing_all": num_missing_all,
                "numbers_in_claim": " | ".join(r.numbers_in_claim),
                "matched_numbers_pdf": " | ".join(r.matched_numbers_pdf),
                "matched_numbers_xlsx": " | ".join(r.matched_numbers_xlsx),
                "missing_numbers_all": " | ".join(r.missing_numbers_all),
                "num_conflict_pdf": int(bool(r.num_conflict_pdf)),
                "num_conflict_xlsx": int(bool(r.num_conflict_xlsx)),

                # claim
                "claim_text": r.claim_text,

                # top evidence (pdf)
                "pdf_top_page": e0.page if e0 else "",
                "pdf_top_chunk": e0.chunk_id if e0 else "",
                "pdf_top_score": e0.score if e0 else "",
                "pdf_top_snippet": e0.snippet if e0 else "",

                # top evidence (xlsx)
                "xlsx_top_row": x0.page if x0 else "",
                "xlsx_top_chunk": x0.chunk_id if x0 else "",
                "xlsx_top_score": x0.score if x0 else "",
                "xlsx_top_snippet": x0.snippet if x0 else "",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = safe_write_csv(df, out_dir / "report.csv")
    xlsx_path = safe_write_excel(df, out_dir / "report.xlsx")

    # HTML (Professional report for credit officers)
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"report_{ts}.html"

    # Build UI payload (keep it compact but decision-ready)
    ui_rows: List[Dict] = []
    for r in results:
        # evidence lists with highlighted snippets
        pdf_evs = []
        for ev in getattr(r, "evidence", []) or []:
            pdf_evs.append(
                {
                    "page": ev.page,
                    "chunk_id": ev.chunk_id,
                    "score": float(ev.score),
                    "snippet_html": highlight_snippet(ev.snippet, r.claim_text),
                    "snippet_raw": ev.snippet,
                }
            )
        xlsx_evs = []
        for ev in getattr(r, "evidence_xlsx", []) or []:
            xlsx_evs.append(
                {
                    "row": ev.page,
                    "chunk_id": ev.chunk_id,
                    "score": float(ev.score),
                    "snippet_html": highlight_snippet(ev.snippet, r.claim_text),
                    "snippet_raw": ev.snippet,
                }
            )

        ui_rows.append(
            {
                "id": r.claim_id,
                "claim": r.claim_text,

                # statuses (raw + ko)
                "overall_raw": r.status_overall,
                "pdf_raw": r.status_pdf,
                "xlsx_raw": r.status_xlsx,
                "overall": STATUS_KO.get(r.status_overall, r.status_overall),
                "pdf": STATUS_KO.get(r.status_pdf, r.status_pdf),
                "xlsx": STATUS_KO.get(r.status_xlsx, r.status_xlsx),

                # scores
                "score_overall": float(r.best_score),
                "score_pdf": float(r.best_score_pdf),
                "score_xlsx": float(r.best_score_xlsx),

                # feature scores
                "sim_pdf": float(getattr(r, "sim_best_pdf", 0.0)),
                "sim_xlsx": float(getattr(r, "sim_best_xlsx", 0.0)),
                "cov_pdf": float(getattr(r, "coverage_pdf", 0.0)),
                "cov_xlsx": float(getattr(r, "coverage_xlsx", 0.0)),
                "prox_pdf": float(getattr(r, "proximity_pdf", 0.0)),
                "prox_xlsx": float(getattr(r, "proximity_xlsx", 0.0)),
                "num_pdf": float(getattr(r, "big_num_recall_pdf", 0.0)),
                "num_xlsx": float(getattr(r, "big_num_recall_xlsx", 0.0)),
                "exact_big_pdf": int(getattr(r, "exact_big_num_matches_pdf", 0) or 0),
                "exact_big_xlsx": int(getattr(r, "exact_big_num_matches_xlsx", 0) or 0),
                "unit_pdf": float(getattr(r, "unit_match_pdf", 0.0)),
                "unit_xlsx": float(getattr(r, "unit_match_xlsx", 0.0)),
                "dir_pdf": float(getattr(r, "direction_score_pdf", 1.0)),
                "dir_xlsx": float(getattr(r, "direction_score_xlsx", 1.0)),
                "dir_conflict_pdf": bool(getattr(r, "direction_conflict_pdf", False)),
                "dir_conflict_xlsx": bool(getattr(r, "direction_conflict_xlsx", False)),
                "num_conflict_pdf": bool(getattr(r, "num_conflict_pdf", False)),
                "num_conflict_xlsx": bool(getattr(r, "num_conflict_xlsx", False)),

                # numbers
                "numbers": getattr(r, "numbers_in_claim", []) or [],
                "missing_numbers": getattr(r, "missing_numbers_all", []) or [],
                "num_total": len(getattr(r, "numbers_in_claim", []) or []),
                "num_missing": len(getattr(r, "missing_numbers_all", []) or []),

                # evidence payload
                "pdf_evidence": pdf_evs,
                "xlsx_evidence": xlsx_evs,
            }
        )

    payload_json = json.dumps(
        {
            "meta": {
                "md": str(md_path),
                "pdf": str(pdf_path),
                "xlsx": [str(p) for p in xlsx_paths_loaded],
                "claims": len(results),
                "top_k": top_k,
            },
            "rows": ui_rows,
        },
        ensure_ascii=False,
    )

    # "전체내용보기" 시트용 원문 로드 (확인리포트와 동일한 MD 파일 사용)
    result_report_path = md_path
    try:
        raw_bytes = result_report_path.read_bytes()
        try:
            result_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                result_text = raw_bytes.decode("cp949")
            except UnicodeDecodeError:
                result_text = raw_bytes.decode("euc-kr", errors="replace")
    except Exception:
        result_text = ""

    def text_to_report_html(text: str) -> str:
        # lightweight pretty renderer for plain text
        t = normalize_invisibles(text or "").strip()
        if not t:
            return "<div class='small'>표시할 내용이 없습니다.</div>"

        lines = [ln.rstrip() for ln in t.splitlines()]
        out_parts: List[str] = []

        def esc(x: str) -> str:
            return html_escape(x)

        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if not ln:
                i += 1
                continue

            # section heading like "1. ..."
            if re.match(r"^\d+\.\s+\S+", ln):
                out_parts.append(f"<h2 class='doc-h2'>{esc(ln)}</h2>")
                i += 1
                continue

            # bracket heading like "[...]" or "【...】"
            if (ln.startswith("[") and ln.endswith("]")) or (ln.startswith("【") and ln.endswith("】")):
                out_parts.append(f"<h3 class='doc-h3'>{esc(ln)}</h3>")
                i += 1
                continue

            # bullet lines (o / - / •)
            if re.match(r"^(o|\-|•)\s+", ln, flags=re.IGNORECASE):
                items: List[str] = []
                while i < len(lines):
                    x = lines[i].strip()
                    if not x or not re.match(r"^(o|\-|•)\s+", x, flags=re.IGNORECASE):
                        break
                    items.append(re.sub(r"^(o|\-|•)\s+", "", x, flags=re.IGNORECASE))
                    i += 1
                out_parts.append("<ul class='doc-ul'>" + "".join([f"<li>{esc(it)}</li>" for it in items]) + "</ul>")
                continue

            # tab-separated table-ish block
            if "\t" in ln:
                rows_block: List[str] = []
                while i < len(lines) and lines[i].strip() and ("\t" in lines[i]):
                    rows_block.append(lines[i].rstrip("\n"))
                    i += 1
                table_rows: List[str] = []
                for ridx, row in enumerate(rows_block):
                    cols = [c.strip() for c in row.split("\t")]
                    tag = "th" if ridx == 0 else "td"
                    tds = "".join([f"<{tag}>{esc(c)}</{tag}>" for c in cols])
                    table_rows.append(f"<tr>{tds}</tr>")
                out_parts.append("<div class='doc-table-wrap'><table class='doc-table'>" + "".join(table_rows) + "</table></div>")
                continue

            # paragraph (consume until blank)
            paras = [ln]
            i += 1
            while i < len(lines) and lines[i].strip():
                paras.append(lines[i].rstrip())
                i += 1
            out_parts.append("<p class='doc-p'>" + esc(" ".join(paras)) + "</p>")

        return "\n".join(out_parts)

    full_report_html = text_to_report_html(result_text)

    # HTML (client-rendered 2-panel report)
    html: List[str] = []
    html.append("<!doctype html><html><head><meta charset='utf-8'/>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    html.append("<title>Hallucination Check Report</title>")
    html.append("<style>")
    html.append(r"""
:root{
  --bg:#0b1220;
  --card:#101a2e;
  --panel:#0f172a;
  --text:#e5e7eb;
  --muted:#a3a3a3;
  --line:rgba(255,255,255,0.09);
  --good:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
  --pill:#111827;
  --shadow: 0 10px 25px rgba(0,0,0,0.35);
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

*{ box-sizing:border-box; }
body{
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans KR", Arial, sans-serif;
  color: var(--text);
  background: radial-gradient(1200px 600px at 10% 0%, rgba(59,130,246,0.18), transparent 55%),
              radial-gradient(1000px 700px at 90% 10%, rgba(245,158,11,0.14), transparent 55%),
              var(--bg);
}
a{ color:#93c5fd; text-decoration:none; }
a:hover{ text-decoration:underline; }
.container{ max-width: 1400px; margin: 0 auto; padding: 18px; }

.header{
  display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; justify-content:space-between;
  padding: 18px; border:1px solid var(--line); border-radius: 16px; background: rgba(16,26,46,0.75);
  box-shadow: var(--shadow);

  /* Keep the big header box visible while scrolling */
  position: sticky;
  top: var(--tabs-h, 0px);
  z-index: 9998;
  backdrop-filter: blur(8px);
}
.h-title{ font-size: 28px; font-weight: 800; letter-spacing: -0.02em; margin:0; }
.h-sub{ font-size:12px; color: var(--muted); margin-top: 6px; line-height:1.45;}
.meta-grid{ display:grid; grid-template-columns: 1fr; gap:6px; font-size:12px; color: #cbd5e1; }
.meta-grid span{ color: var(--muted); margin-right:6px; }

.actions{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.scorebox{ margin-left:auto; display:flex; flex-direction:column; align-items:flex-end; gap:6px; min-width:360px; }
#docGradeBadge{ transform: scale(1.08); transform-origin: top right; }

.btn{
  padding: 9px 12px; border-radius: 12px; border: 1px solid var(--line);
  background: rgba(17,24,39,0.75); color: var(--text); cursor:pointer;
}
.btn:hover{ background: rgba(17,24,39,0.95); }
.btn.primary{ background: rgba(59,130,246,0.22); border-color: rgba(59,130,246,0.35); }
.btn.danger{ background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.35); }
.btn:disabled{ opacity:0.5; cursor:not-allowed; }

/* sheet tabs */
.sheet-tabs{
  position: sticky;
  top: 0;
  z-index: 9999;
  display:flex;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(15,23,42,0.92);
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}
.sheet-tab{
  padding: 9px 12px;
  border-radius: 12px;
  border: 1px solid var(--line);
  background: rgba(17,24,39,0.6);
  color: var(--text);
  cursor:pointer;
  font-weight: 800;
}
.sheet-tab.active{
  background: rgba(59,130,246,0.25);
  border-color: rgba(59,130,246,0.40);
}
.sheet{ margin-top: 14px; }

/* KPI row in 확인리포트 */
#sheetReport .kpis{
  position: sticky;
  top: calc(var(--tabs-h, 0px) + var(--header-h, 0px) + 10px);
  z-index: 9997;
  backdrop-filter: blur(8px);
}

/* Filters row in 확인리포트 */

/* "전체내용보기" report styling */
.pulse{ animation: pulse 1.2s ease-in-out 1; }
@keyframes pulse{ 0%{ box-shadow:0 0 0 0 rgba(239,68,68,0.0);} 30%{ box-shadow:0 0 0 6px rgba(239,68,68,0.25);} 100%{ box-shadow:0 0 0 0 rgba(239,68,68,0.0);} }

.must-review{
  color:#fecaca;
  background: rgba(239,68,68,0.12);
  border-bottom: 1px dashed rgba(239,68,68,0.65);
  padding: 1px 2px;
  border-radius: 4px;
  cursor:pointer;
}
.must-review:hover{ background: rgba(239,68,68,0.18); }

/* decision styling in 전체내용보기 (근거 부족 문장에 적용) */
.dec-confirmed{
  color:#bfdbfe;
  background: rgba(59,130,246,0.14);
  border-bottom: 1px dashed rgba(59,130,246,0.60);
  padding: 1px 2px;
  border-radius: 4px;
  cursor:pointer;
}
.dec-confirmed:hover{ background: rgba(59,130,246,0.20); }

.dec-edit{
  color:#fde68a;
  background: rgba(245,158,11,0.16);
  border-bottom: 1px dashed rgba(245,158,11,0.70);
  padding: 1px 2px;
  border-radius: 4px;
  cursor:pointer;
}
.dec-edit:hover{ background: rgba(245,158,11,0.22); }

.dec-delete{
  color:#94a3b8;
  opacity: 0.85;
  text-decoration: line-through;
  text-decoration-thickness: 2px;
  text-decoration-color: rgba(148,163,184,0.9);
  cursor:pointer;
}
.dec-delete:hover{ opacity: 1.0; }

.doc{
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(16,26,46,0.62);
  box-shadow: var(--shadow);
}
.doc-head{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 14px;
}
.doc-title{ font-size: 22px; font-weight: 900; letter-spacing:-0.02em; }
.doc-meta{ color: var(--muted); font-size: 12px; }
.doc-h2{ margin: 18px 0 10px; font-size: 18px; font-weight: 900; }
.doc-h3{ margin: 14px 0 8px; font-size: 14px; font-weight: 900; color:#cbd5e1; }
.doc-p{ margin: 10px 0; color:#e5e7eb; line-height: 1.75; }
.doc-ul{ margin: 8px 0 12px 18px; }
.doc-ul li{ margin: 6px 0; color:#e5e7eb; line-height:1.65; }
.doc-table-wrap{ overflow:auto; border:1px solid rgba(255,255,255,0.10); border-radius: 14px; margin: 10px 0 14px; }
.doc-table{ width:100%; border-collapse: collapse; min-width: 640px; }
.doc-table th, .doc-table td{
  padding: 10px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  font-size: 13px;
  white-space: nowrap;
}
.doc-table th{ color:#cbd5e1; background: rgba(15,23,42,0.9); position: sticky; top: 0; }

/* report attention highlight */
.attn{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(239,68,68,0.35);
  background: rgba(239,68,68,0.14);
  color: #fecaca;
  font-size: 12px;
  font-weight: 800;
  cursor:pointer;
}
.queue-table tbody tr.needs-attn{
  box-shadow: inset 4px 0 0 rgba(239,68,68,0.55);
  background: rgba(239,68,68,0.06);
}

/* full sheet attention list */
.attn-panel{
  margin: 14px 0 16px;
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(239,68,68,0.22);
  background: rgba(239,68,68,0.06);
}
.attn-panel .title{
  font-weight: 900;
  margin-bottom: 8px;
  color: #fecaca;
}
.attn-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 10px;
}
.attn-card{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  background: rgba(16,26,46,0.65);
  padding: 12px;
  cursor:pointer;
}
.attn-card:hover{ background: rgba(16,26,46,0.82); }
.attn-card .meta{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.attn-card .snippet{ margin-top: 8px; color:#e5e7eb; font-size:13px; line-height:1.55; }
.attn-card .sub{ margin-top: 8px; color: var(--muted); font-size: 12px; }
@media (max-width: 1100px){
  .attn-grid{ grid-template-columns: 1fr; }
}

.kpis{
  display:grid;
  grid-template-columns: repeat(6, minmax(140px, 1fr));
  gap: 10px;
  margin-top: 14px;
}
.grade-wrap{
  margin-top: 14px;
  display:flex;
  align-items:center;
  justify-content:flex-start;
  gap: 10px;
}
.grade-badge{
  display:inline-flex;
  align-items:baseline;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: rgba(16,26,46,0.78);
  box-shadow: 0 10px 22px rgba(0,0,0,0.25);
}
.grade-letter{
  font-size: 24px;
  font-weight: 1000;
  letter-spacing: -0.02em;
}
.grade-score{
  font-size: 14px;
  color: #cbd5e1;
  font-weight: 800;
}
.grade-desc{
  font-size: 12px;
  color: var(--muted);
  margin-left: 6px;
}
.grade-badge.gA{ border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.10); }
.grade-badge.gB{ border-color: rgba(34,197,94,0.30); background: rgba(34,197,94,0.08); }
.grade-badge.gC{ border-color: rgba(245,158,11,0.30); background: rgba(245,158,11,0.08); }
.grade-badge.gF{ border-color: rgba(239,68,68,0.30); background: rgba(239,68,68,0.08); }

.kpi{
  padding: 14px; border-radius: 14px; border: 1px solid var(--line);
  background: rgba(16,26,46,0.72);
  box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}
.kpi .label{ font-size:11px; color: var(--muted); }
.kpi .value{ font-size: 22px; font-weight: 800; margin-top: 6px; }
.kpi .hint{ font-size:11px; color:#cbd5e1; margin-top: 4px; }

.filters{
  margin-top: 14px;
  padding: 14px; border-radius: 16px; border: 1px solid var(--line);
  background: rgba(16,26,46,0.55);
  display:flex; flex-wrap:wrap; gap:10px; align-items:center;

  /* Keep filters visible while scrolling in "확인리포트" */
  position: sticky;
  top: calc(var(--tabs-h, 0px) + var(--header-h, 0px) + var(--kpis-h, 0px) + 20px);
  z-index: 9996;
  backdrop-filter: blur(8px);
}

/* Panel headers should sit at the very top INSIDE each panel */
#sheetReport .panel .p-head{
  position: sticky;
  top: 0;
  z-index: 30;
  background: rgba(11,18,36,0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid rgba(255,255,255,0.10);
}
.filters input, .filters select{
  padding: 10px 10px; border-radius: 12px; border: 1px solid var(--line);
  background: rgba(15,23,42,0.7); color: var(--text);
  outline:none;
}
.filters input{ min-width: 260px; }
.filters .spacer{ flex: 1; }
.small{ font-size: 12px; color: var(--muted); }

.main{
  margin-top: 14px;
  display:grid;
  grid-template-columns: 520px 1fr;
  gap: 12px;
  min-height: calc(100vh - 280px);
}
.panel{
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(16,26,46,0.55);
  box-shadow: var(--shadow);
  overflow:hidden;
  display:flex;
  flex-direction:column;
}
.panel .p-head{
  padding: 12px 14px;
  border-bottom: 1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between;
}
.panel .p-head .title{ font-weight: 800; }
.panel .p-body{ padding: 0; overflow:auto; height:100%; }

/* Keep detail panel below sticky stack (tabs/header/kpi/filters) */
#detailPanel{
  position: sticky;
  top: calc(var(--report-stack-h, 0px) + 10px);
  align-self: start;
  max-height: calc(100vh - var(--report-stack-h, 0px) - 24px);
  scroll-margin-top: calc(var(--report-stack-h, 0px) + 10px);
}

/* When scrolling to the queue panel, also avoid being hidden by sticky stack */
#sheetReport .main{
  scroll-margin-top: calc(var(--report-stack-h, 0px) + 10px);
}

.queue-table{ width:100%; border-collapse: collapse; }
.queue-table thead th{
  position: sticky; top:0;
  background: rgba(15,23,42,0.95);
  border-bottom: 1px solid var(--line);
  padding: 10px 10px; text-align:left; font-size: 12px; color:#cbd5e1;
}
.queue-table tbody td{
  border-bottom: 1px solid rgba(255,255,255,0.06);
  padding: 10px 10px; font-size: 13px; vertical-align: top;
}
.queue-table tbody tr{ cursor:pointer; }
.queue-table tbody tr:hover{ background: rgba(59,130,246,0.10); }
.queue-table tbody tr.selected{ background: rgba(59,130,246,0.18); }
.queue-table tbody tr.confirmed{ background: rgba(34,197,94,0.15); border-left: 3px solid rgba(34,197,94,0.6); }
.queue-table tbody tr.confirmed:hover{ background: rgba(34,197,94,0.20); }
.queue-table tbody tr.needs-edit{ background: rgba(245,158,11,0.14); border-left: 3px solid rgba(245,158,11,0.60); }
.queue-table tbody tr.needs-edit:hover{ background: rgba(245,158,11,0.20); }
.queue-table tbody tr.deleted{ background: rgba(239,68,68,0.12); border-left: 3px solid rgba(239,68,68,0.60); }
.queue-table tbody tr.deleted:hover{ background: rgba(239,68,68,0.18); }
.queue-table tbody tr.hold{ background: rgba(148,163,184,0.12); border-left: 3px solid rgba(148,163,184,0.55); }
.queue-table tbody tr.hold:hover{ background: rgba(148,163,184,0.16); }

/* Modal popup */
.modal-overlay{
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.75);
  display: flex; align-items: center; justify-content: center;
  z-index: 10000;
}
.modal-content{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 20px;
  max-width: 600px;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: var(--shadow);
}
.modal-header{
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 12px;
}
.modal-title{ font-size: 18px; font-weight: 800; }
.modal-close{
  background: none; border: none; color: var(--text);
  font-size: 24px; cursor: pointer; padding: 0; width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 8px;
}
.modal-close:hover{ background: rgba(255,255,255,0.1); }
.modal-body{ font-size: 14px; line-height: 1.6; }
.modal-body h4{ margin-top: 16px; margin-bottom: 8px; color: var(--text); }
.modal-body code{ background: rgba(15,23,42,0.7); padding: 2px 6px; border-radius: 4px; font-family: var(--mono); }

/* Batch action buttons */
.batch-actions{
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  background: rgba(15,23,42,0.5);
}
.batch-actions select{
  padding: 6px 10px; border-radius: 8px; border: 1px solid var(--line);
  background: rgba(15,23,42,0.7); color: var(--text); font-size: 12px;
}
.batch-actions .small{ margin-left: auto; color: var(--muted); }

.pill{
  display:inline-flex; align-items:center; gap:6px;
  padding: 4px 10px; border-radius: 999px;
  background: rgba(17,24,39,0.85);
  border: 1px solid rgba(255,255,255,0.10);
  font-size: 12px;
}
.pill.good{ border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.12); }
.pill.warn{ border-color: rgba(245,158,11,0.45); background: rgba(245,158,11,0.12); }
.pill.bad{ border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.12); }

.tag{
  display:inline-flex;
  padding: 2px 8px;
  font-size: 11px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(2,6,23,0.55);
  color:#cbd5e1;
}
.tag.bad{ border-color: rgba(239,68,68,0.35); }
.tag.warn{ border-color: rgba(245,158,11,0.35); }
.tag.info{ border-color: rgba(59,130,246,0.35); }
.tag.good{ border-color: rgba(34,197,94,0.35); }

.score{
  font-family: var(--mono);
  font-weight: 700;
  font-size: 14px;
}
.bar{
  width: 100%;
  height: 8px;
  background: rgba(255,255,255,0.08);
  border-radius: 999px;
  overflow:hidden;
  margin-top: 6px;
}
.bar > div{
  height: 100%;
  width: 0%;
  background: rgba(59,130,246,0.8);
}

.detail{
  padding: 14px;
}
.claim-box{
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(2,6,23,0.45);
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.claim-content{ flex: 1; }
.claim-title{ font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.claim-text{ font-size: 15px; line-height: 1.55; font-weight: 650; }
.decision-select-box{
  padding: 8px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(15,23,42,0.7);
  min-width: 140px;
}
.decision-select-box select{
  width: 100%;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(2,6,23,0.55);
  color: var(--text);
  font-size: 12px;
  outline: none;
}

.grid2{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.card{
  padding: 12px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(2,6,23,0.45);
}
.card h3{ margin:0 0 8px 0; font-size: 13px; }
.card .row{ display:flex; justify-content:space-between; align-items:center; margin-top:6px; font-size: 12px; color:#cbd5e1; }
.card .row span{ color: var(--muted); }
.card .mono{ font-family: var(--mono); }

.evidence-tabs{ display:flex; gap:8px; margin-top: 12px; }
.tab{ padding: 8px 10px; border-radius: 12px; border:1px solid var(--line); background: rgba(15,23,42,0.7); cursor:pointer; font-size: 12px;}
.tab.active{ background: rgba(59,130,246,0.20); border-color: rgba(59,130,246,0.35); }

.evidence-card{
  margin-top: 10px;
  padding: 12px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(2,6,23,0.45);
}
.e-head{ display:flex; justify-content:space-between; align-items:center; gap:10px; }
.e-head .meta{ font-size: 12px; color:#cbd5e1; }
.e-head .meta b{ font-family: var(--mono); font-weight:700; }
.snippet{
  margin-top: 10px;
  font-size: 12px; line-height: 1.55;
  color:#e5e7eb;
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 10px;
  border-radius: 12px;
}
.snippet b{ font-weight: 900; color: #fbbf24; background: rgba(251,191,36,0.15); padding: 2px 4px; border-radius: 4px; }

.review-box{
  margin-top: 12px;
  padding: 12px;
  border-radius: 16px;
  border: 1px dashed rgba(255,255,255,0.20);
  background: rgba(2,6,23,0.35);
}
.review-box label{ font-size: 12px; color: var(--muted); display:block; margin-bottom:6px; }
.review-box select, .review-box textarea{
  width: 100%;
  padding: 10px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(15,23,42,0.7);
  color: var(--text);
  outline:none;
}
.review-box textarea{ min-height: 92px; resize: vertical; margin-top: 8px; }
.review-actions{ display:flex; gap:8px; margin-top: 10px; }
.footer{
  margin-top: 12px;
  padding: 12px 6px;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.5;
}
@media (max-width: 1100px){
  .main{ grid-template-columns: 1fr; }
  .kpis{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}
""")
    html.append("</style></head><body>")
    html.append("<div class='container'>")

    # sheet tabs (default: full view)
    html.append("<div class='sheet-tabs'>")
    html.append("<button class='sheet-tab active' id='sheetTabFull'>전체내용보기</button>")
    html.append("<button class='sheet-tab' id='sheetTabReport'>확인리포트</button>")
    html.append("</div>")

    # header
    html.append("<div class='header'>")
    html.append("<div>")
    html.append("<div class='h-title'>Hallucination Check Report</div>")
    html.append("<div class='h-sub'>여신심사 담당자/관리자 공유용 검증 보고서 • 검토 큐(좌)에서 항목 선택 → 근거 비교(우)에서 확인/조치</div>")
    html.append("</div>")
    html.append("<div class='meta-grid'>")
    html.append(f"<div><span>MD</span>{html_escape(str(md_path))}</div>")
    html.append(f"<div><span>PDF</span>{html_escape(str(pdf_path))}</div>")
    if xlsx_paths_loaded:
        for p in xlsx_paths_loaded:
            html.append(f"<div><span>XLSX</span>{html_escape(str(p))}</div>")
    html.append(f"<div><span>Claims</span>{len(results)} / <span>Top-K</span>{top_k}</div>")
    html.append("</div>")
    html.append("<div class='actions'>")
    html.append("<button class='btn primary' id='exportDecisionsBtn'>검토결과 CSV 내보내기</button>")
    html.append("<button class='btn' id='exportViewBtn'>현재목록 CSV 내보내기</button>")
    html.append("<button class='btn' id='printBtn'>인쇄/저장(PDF)</button>")
    html.append("<button class='btn danger' id='clearDecisionsBtn'>검토결과 초기화</button>")
    html.append("<button class='btn' id='reviewScoreBtn'>검토 결과 확인</button>")
    html.append("<div class='scorebox'>")
    html.append("<div class='grade-badge' id='docGradeBadge'><span class='grade-letter'>-</span><span class='grade-score'>0/100</span><span class='grade-desc'>문서품질점수</span></div>")
    html.append("<div class='small' id='reviewScoreOut' style='min-width:360px; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'></div>")
    html.append("</div>")  # scorebox
    html.append("</div>")
    html.append("</div>")  # header

    # --- Sheet: Full content view (확인리포트와 동일 데이터 소스) ---
    html.append("<div class='sheet' id='sheetFull'>")
    html.append("<div class='doc'>")
    html.append("<div class='doc-head'>")
    html.append("<div class='doc-title'>전체내용보기</div>")
    html.append("<div class='doc-meta'>")
    html.append(f"검증 대상 원문: {html_escape(str(result_report_path))} • 근거: PDF + XLSX(복수) • 문장 클릭 시 확인리포트로 이동")
    html.append("</div>")
    html.append("</div>")
    html.append(full_report_html)
    html.append("</div>")
    html.append("</div>")

    # --- Sheet: Report (existing UI) ---
    html.append("<div class='sheet' id='sheetReport' style='display:none'>")

    # KPI placeholders (keep existing)
    
    html.append("<div class='kpis' id='kpis'>")
    for _ in range(6):
        html.append("<div class='kpi'><div class='label'>-</div><div class='value'>-</div><div class='hint'>-</div></div>")
    html.append("</div>")

    # filters
    html.append("<div class='filters'>")
    html.append("<input id='q' placeholder='Claim 검색 (키워드/숫자)'>")
    html.append("<input id='idFilter' placeholder='ID 필터 예: 1-20, 35, 40-45'>")
    html.append("<select id='overallFilter'>"
                "<option value='ALL'>통합: 전체</option>"
                "<option value='SUPPORTED'>통합: 근거확인</option>"
                "<option value='PARTIAL'>통합: 부분확인</option>"
                "<option value='UNSUPPORTED'>통합: 근거미확인</option>"
                "</select>")
    html.append("<select id='riskFilter'>"
                "<option value='ALL'>리스크: 전체</option>"
                "<option value='NUM_MISSING'>숫자 누락</option>"
                "<option value='BIG_NUM_MISSING'>큰 숫자 누락(3자리+)</option>"
                "<option value='DISPERSED'>근거 분산</option>"
                "<option value='PDF_ONLY'>PDF만 근거</option>"
                "<option value='JSON_ONLY'>JSON만 근거</option>"
                "<option value='DIR_CONFLICT'>방향성 모순</option>"
                "<option value='UNIT_MISMATCH'>단위 불일치</option>"
                "</select>")
    html.append(                "<select id='decisionFilter'>"
                "<option value='ALL'>검토결정: 전체</option>"
                "<option value='NONE'>미결정</option>"
                "<option value='CONFIRMED'>확인</option>"
                "<option value='EDIT'>수정필요</option>"
                "<option value='DELETE'>삭제</option>"
                "<option value='HOLD'>보류</option>"
                "</select>")
    html.append("<select id='sortBy'>"
                "<option value='id_asc' selected>정렬: ID↑</option>"
                "<option value='priority'>정렬: 우선순위</option>"
                "<option value='score_desc'>정렬: 점수↓</option>"
                "<option value='score_asc'>정렬: 점수↑</option>"
                "</select>")
    html.append("<div class='spacer'></div>")
    html.append("<div class='small' id='countInfo'>표시 0 / 전체 0</div>")
    html.append("</div>")

    # main layout
    html.append("<div class='main'>")

    # queue panel
    html.append("<div class='panel'>")
    html.append("<div class='p-head'><div class='title'>검토 큐</div><div class='small'>우선순위 기반 • 행 클릭</div></div>")
    html.append("<div class='batch-actions' id='batchActions' style='display:none;'>")
    html.append("<select id='batchDecision'>")
    html.append("<option value='CONFIRMED'>확인</option>")
    html.append("<option value='EDIT'>수정필요</option>")
    html.append("<option value='DELETE'>삭제</option>")
    html.append("<option value='HOLD'>보류</option>")
    html.append("<option value='NONE'>미결정</option>")
    html.append("</select>")
    html.append("<button class='btn primary' id='batchApplyBtn'>선택 항목 적용</button>")
    html.append("<button class='btn' id='batchSelectAllBtn'>전체 선택</button>")
    html.append("<button class='btn' id='batchDeselectAllBtn'>전체 해제</button>")
    html.append("<div class='small' id='batchCount'>선택: 0건</div>")
    html.append("</div>")
    html.append("<div class='p-body'>")
    html.append("<table class='queue-table' id='queueTable'>")
    html.append("<thead><tr><th style='width:30px'><input type='checkbox' id='selectAllCheckbox'></th><th style='width:60px'>ID</th><th style='width:130px'>판정</th><th style='width:90px'>점수</th><th>리스크/Claim</th></tr></thead>")
    html.append("<tbody></tbody></table>")
    html.append("</div></div>")

    # detail panel
    html.append("<div class='panel' id='detailPanel'>")
    html.append("<div class='p-head'><div class='title'>상세 검토</div><div class='small' id='detailHint'>좌측에서 항목을 선택하세요</div></div>")
    html.append("<div class='p-body'><div class='detail' id='detailPane'></div></div>")
    html.append("</div>")

    html.append("</div>")  # main

    # footer
    html.append("<div class='footer'>"
                "<b>운영 가이드</b><br/>"
                "• <span class='tag warn'>근거 분산</span>: 용어/숫자가 문서 여러 위치에 흩어져 있을 수 있습니다. PDF/JSON 탭의 'More evidence'를 확인하세요.<br/>"
                "• <span class='tag bad'>숫자 누락</span>/<span class='tag bad'>큰 숫자 누락</span>: 재무수치/비율 오류 가능성이 높습니다. 원문 근거를 기준으로 문장을 수정 권장.<br/>"
                "• 본 보고서는 자동 검증 보조 도구이며, 최종 판단은 담당자의 검토를 전제로 합니다."
                "</div>")

    html.append("</div>")  # sheetReport

    # embedded payload
    html.append("<script id='payload' type='application/json'>")
    html.append(payload_json)
    html.append("</script>")

    # JS app
    html.append("<script>")
    html.append(r"""
(function(){
  const payload = JSON.parse(document.getElementById('payload').textContent);
  const rows = payload.rows || [];
  const meta = payload.meta || {};
  const el = (id)=>document.getElementById(id);

  // --- sticky offsets (tabs/header/kpis/filters) ---
  function updateStickyMetrics(){
    const tabs = document.querySelector('.sheet-tabs');
    const header = document.querySelector('.header');
    const kpis = document.querySelector('#sheetReport .kpis');
    const filters = document.querySelector('#sheetReport .filters');

    const tabsH = tabs ? Math.ceil(tabs.getBoundingClientRect().height || 0) : 0;
    const headerH = header ? Math.ceil(header.getBoundingClientRect().height || 0) : 0;

    // Only count KPI/filters when report sheet is visible
    const isReportVisible = (()=>{
      const rep = el('sheetReport');
      if(!rep) return false;
      // display==none when full view
      return rep.style.display !== 'none';
    })();

    const kpisH = (isReportVisible && kpis) ? Math.ceil(kpis.getBoundingClientRect().height || 0) : 0;
    const filtersH = (isReportVisible && filters) ? Math.ceil(filters.getBoundingClientRect().height || 0) : 0;

    document.documentElement.style.setProperty('--tabs-h', `${tabsH}px`);
    document.documentElement.style.setProperty('--header-h', `${headerH}px`);
    document.documentElement.style.setProperty('--kpis-h', `${kpisH}px`);
    document.documentElement.style.setProperty('--filters-h', `${filtersH}px`);

    // Bottom edge of sticky stack (tabs+header+kpis+filters), aligned with CSS offsets (+20)
    const reportStack = tabsH + headerH + kpisH + filtersH + 20;
    document.documentElement.style.setProperty('--report-stack-h', `${reportStack}px`);
  }
  window.addEventListener('resize', updateStickyMetrics);
  updateStickyMetrics();

  function getReportStackH(){
    const v = getComputedStyle(document.documentElement).getPropertyValue('--report-stack-h') || '0';
    const n = parseFloat(String(v).replace('px','')) || 0;
    return n;
  }

  function scrollToElementTop(elm, extra=10){
    if(!elm) return;
    const stack = getReportStackH();
    const rect = elm.getBoundingClientRect();
    const y = rect.top + window.scrollY - stack - extra;
    window.scrollTo({ top: Math.max(0, y), behavior: "smooth" });
  }

  function showSheet(which){
    const full = el("sheetFull");
    const rep = el("sheetReport");
    const tabFull = el("sheetTabFull");
    const tabRep = el("sheetTabReport");
    if(!full || !rep || !tabFull || !tabRep) return;

    if(which === "report"){
      full.style.display = "none";
      rep.style.display = "";
      tabFull.classList.remove("active");
      tabRep.classList.add("active");
      // recalc sticky heights when switching to report
      updateStickyMetrics();
    }else{
      full.style.display = "";
      rep.style.display = "none";
      tabRep.classList.remove("active");
      tabFull.classList.add("active");
      updateStickyMetrics();
    }
  }

  // --- decisions (localStorage) ---
  const LS_KEY = "hallucination_review_decisions_v1";
  function loadDecisions(){
    try{ return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }catch(e){ return {}; }
  }
  function saveDecisions(map){
    localStorage.setItem(LS_KEY, JSON.stringify(map));
  }
  function getDecision(id){
    const map = loadDecisions();
    return map[String(id)] || {decision:"NONE", note:""};
  }
  function setDecision(id, decision, note){
    const map = loadDecisions();
    map[String(id)] = {decision, note, updated_at: new Date().toISOString()};
    saveDecisions(map);
  }
  function clearDecisions(){
    localStorage.removeItem(LS_KEY);
  }

  // --- helpers ---
  function statusPill(raw){
    if(raw==="SUPPORTED") return `<span class="pill good">✅ 근거확인</span>`;
    if(raw==="PARTIAL") return `<span class="pill warn">⚠️ 부분확인</span>`;
    return `<span class="pill bad">❌ 근거미확인</span>`;
  }
  function statusLabel(raw){
    if(raw==="SUPPORTED") return "근거확인";
    if(raw==="PARTIAL") return "부분확인";
    return "근거미확인";
  }
  function decisionPill(d){
    if(d==="CONFIRMED") return `<span class="tag good">확인</span>`;
    if(d==="EDIT") return `<span class="tag warn">수정필요</span>`;
    if(d==="DELETE") return `<span class="tag bad">삭제</span>`;
    if(d==="HOLD") return `<span class="tag info">보류</span>`;
    return `<span class="tag">미결정</span>`;
  }
  function decisionLabel(d){
    if(d==="CONFIRMED") return "확인";
    if(d==="EDIT") return "수정";
    if(d==="DELETE") return "삭제";
    if(d==="HOLD") return "보류";
    return "미결정";
  }
  function fmt(x){
    if(x===null || x===undefined || Number.isNaN(x)) return "-";
    return (Math.round(x*1000)/1000).toFixed(3);
  }
  function scoreBar(score){
    const w = Math.max(0, Math.min(100, score*100));
    return `<div class="score">${fmt(score)}</div><div class="bar"><div style="width:${w}%"></div></div>`;
  }
  function hasNumbers(r){ return (r.num_total||0) > 0; }
  function anyEvidence(r){
    return (r.pdf_evidence && r.pdf_evidence.length>0) || (r.xlsx_evidence && r.xlsx_evidence.length>0);
  }

  // risk tags (decision-ready)
  function riskTags(r){
    const tags = [];
    const nums = (r.numbers||[]);
    const missing = (r.missing_numbers||[]);
    const missingCount = missing.length;

    if(nums.length>0 && missingCount>0){
      tags.push({key:"NUM_MISSING", label:"숫자 누락", tone:"bad"});
    }

    // big number missing: any missing number with >=3 digits (digits only) or comma groups
    const bigMissing = missing.some(n=>{
      const digits = String(n).replace(/[^0-9]/g,"");
      return digits.length>=3;
    });
    if(bigMissing){
      tags.push({key:"BIG_NUM_MISSING", label:"큰 숫자 누락", tone:"bad"});
    }

    // dispersed evidence: high coverage but low proximity (either source)
    const cov = Math.max(r.cov_pdf||0, r.cov_json||0);
    const prox = Math.max(r.prox_pdf||0, r.prox_json||0);
    if(cov>=0.55 && prox<=0.15){
      tags.push({key:"DISPERSED", label:"근거 분산", tone:"warn"});
    }

    // source-only
    if(r.pdf_raw==="SUPPORTED" && r.xlsx_raw!=="SUPPORTED"){
      tags.push({key:"PDF_ONLY", label:"PDF만", tone:"info"});
    }
    if(r.xlsx_raw==="SUPPORTED" && r.pdf_raw!=="SUPPORTED"){
      tags.push({key:"JSON_ONLY", label:"JSON만", tone:"info"});
    }

    // direction conflict
    if(r.dir_conflict_pdf || r.dir_conflict_json){
      tags.push({key:"DIR_CONFLICT", label:"방향성 모순", tone:"bad"});
    }

    // number conflict suspicion (if available)
    if(r.num_conflict_pdf || r.num_conflict_xlsx){
      tags.push({key:"NUM_CONFLICT", label:"숫자 불일치 의심", tone:"bad"});
    }

    // unit mismatch: if unit score low while numbers exist
    if(nums.length>0 && Math.max(r.unit_pdf||1, r.unit_json||1) < 0.5){
      tags.push({key:"UNIT_MISMATCH", label:"단위 불일치", tone:"warn"});
    }

    return tags;
  }

  function needsAttention(r, decision){
    // "확인리포트"에서 사람이 확인해야 할 항목 하이라이트
    decision = decision || "NONE";
    // 사용자가 '확인(CONFIRMED)'으로 최종조치한 건은 더 이상 경고로 보지 않음
    if(decision === "CONFIRMED") return false;
    const st = r.overall_raw || "UNSUPPORTED";
    const tags = riskTags(r).map(t=>t.key);
    if(st !== "SUPPORTED") return true;
    if(tags.includes("NUM_MISSING") || tags.includes("BIG_NUM_MISSING") || tags.includes("DIR_CONFLICT") || tags.includes("UNIT_MISMATCH") || tags.includes("NUM_CONFLICT")) return true;
    return false;
  }

  function jumpToClaim(id){
    // switch to report sheet, select row, show detail
    showSheet("report");
    const r = rows.find(x=>x.id===id);
    if(!r) return;
    selectedId = id;
    renderQueue(currentList);
    renderDetail(r);
    // scroll into view
    const tr = document.querySelector(`#queueTable tbody tr[data-id="${id}"]`);
    if(tr) tr.scrollIntoView({behavior:"smooth", block:"center"});
    // also bring detail panel header fully into view (not hidden by sticky stack)
    const detailPanel = el("detailPanel");
    if(detailPanel) scrollToElementTop(detailPanel, 10);
  }

  function isEvidenceInsufficient(r){
    const st = r.overall_raw || "UNSUPPORTED";
    const tags = riskTags(r).map(t=>t.key);
    const score = (r.score_overall||0);
    // '근거 부족' 기준(확정):
    // 1) 자동판정이 SUPPORTED가 아님(PARTIAL/UNSUPPORTED)
    // 2) 숫자/단위/방향/근거충돌 관련 리스크 태그 존재
    // 3) 점수(score_overall)가 임계치 이하
    const evidenceTagKeys = [
      "NUM_MISSING","BIG_NUM_MISSING","UNIT_MISMATCH","DIR_CONFLICT","NUM_CONFLICT"
    ];
    const hasEvidenceTag = evidenceTagKeys.some(k=>tags.includes(k));
    const lowScore = score <= 0.20; // 임계치(조정 가능)
    const weakStatus = (st !== "SUPPORTED");
    return weakStatus || hasEvidenceTag || lowScore;
  }

  function mustReviewInFull(r, decision){
    // 전체내용보기에서 '미결정인데 반드시 직접 봐야 할' 후보(확정 기준)
    // 조건: 미결정 + (근거 부족 OR 고리스크)
    if((decision||"NONE") !== "NONE") return false;

    const tags = riskTags(r).map(t=>t.key);
    const w = weightOf(r);

    const highRisk =
      (w >= 1.8) ||
      tags.includes("BIG_NUM_MISSING") ||
      tags.includes("DIR_CONFLICT") ||
      tags.includes("NUM_CONFLICT") ||
      tags.includes("UNIT_MISMATCH");

    return isEvidenceInsufficient(r) || highRisk;
  }

  function buildFullAttnList(){
    // 전체내용보기: 근거 부족 문장은 "문장 자체"를 색으로 표시 + 클릭 시 확인리포트로 이동
    // - 미결정: 빨강(기본)
    // - 확인(CONFIRMED): 파랑
    // - 수정필요(EDIT): 노랑
    // - 삭제(DELETE): 취소선
    const root = document.querySelector("#sheetFull .doc");
    if(!root) return;

    // 중복 방지: 기존 하이라이트 제거
    root.querySelectorAll("span.must-review, span.dec-confirmed, span.dec-edit, span.dec-delete").forEach(sp=>{
      const t = document.createTextNode(sp.textContent || "");
      sp.parentNode && sp.parentNode.replaceChild(t, sp);
    });

    const dec = loadDecisions();
    const targets = [];
    for(const r of rows){
      if(!isEvidenceInsufficient(r)) continue;
      const d = dec[String(r.id)] || {decision:"NONE"};
      const decision = (d.decision || "NONE");
      const claim = (r.claim || "").trim();
      if(!claim) continue;

      let cls = "must-review";
      let tip = "클릭하여 확인리포트로 이동";
      if(decision === "CONFIRMED"){ cls = "dec-confirmed"; tip = "확인 처리됨 • 클릭하여 확인리포트로 이동"; }
      else if(decision === "EDIT"){ cls = "dec-edit"; tip = "수정필요 처리됨 • 클릭하여 확인리포트로 이동"; }
      else if(decision === "DELETE"){ cls = "dec-delete"; tip = "삭제 처리됨 • 클릭하여 확인리포트로 이동"; }
      else if(decision === "HOLD"){ cls = "must-review"; tip = "보류 • 클릭하여 확인리포트로 이동"; }
      else { cls = "must-review"; tip = "미결정(확인필요) • 클릭하여 확인리포트로 이동"; }

      targets.push({id: r.id, claim, cls, tip});
    }
    if(targets.length===0) return;

    function htmlEscape(s){
      return String(s)
        .replace(/&/g,"&amp;")
        .replace(/</g,"&lt;")
        .replace(/>/g,"&gt;")
        .replace(/"/g,"&quot;")
        .replace(/'/g,"&#39;");
    }

    // 클릭 이벤트(1회 바인딩)
    if(!window.__mustReviewBound){
      window.__mustReviewBound = true;
      document.addEventListener("click", (e)=>{
        const sp = e.target && e.target.closest ? e.target.closest(".must-review, .dec-confirmed, .dec-edit, .dec-delete") : null;
        if(!sp) return;
        const id = parseInt(sp.dataset.id || "0", 10);
        if(id) jumpToClaim(id);
      });
    }

    const nodes = root.querySelectorAll(".doc-p, .doc-ul li, .doc-table td, .doc-table th, .doc-h2, .doc-h3");
    const marked = new Set();

    // 긴 claim 우선(부분 매칭 방지)
    targets.sort((a,b)=> (b.claim.length - a.claim.length));

    for(const t of targets){
      const escClaim = htmlEscape(t.claim).replace(/\s+/g, " ");
      for(const n of nodes){
        if(marked.has(t.id)) break;
        if(!n || !n.innerHTML) continue;

        // innerHTML은 이미 escape된 텍스트이므로 escape된 claim으로 매칭
        const idx = n.innerHTML.indexOf(escClaim);
        if(idx === -1) continue;

        // 첫 1회만 치환
        n.innerHTML = n.innerHTML.replace(
          escClaim,
          `<span class="${t.cls}" data-id="${t.id}" title="${t.tip}">${escClaim}</span>`
        );
        marked.add(t.id);
        break;
      }
    }
  }

  function recommendedActionOf(r){
    const st = r.overall_raw || "UNSUPPORTED";
    const tags = riskTags(r).map(t=>t.key);
    const hasNum = hasNumberInClaim(r) || hasNumbers(r);

    // rule-based recommendation (simple + deterministic)
    if(tags.includes("DIR_CONFLICT")) return "수정 권장";
    if(tags.includes("BIG_NUM_MISSING")) return "수정 권장";
    if(st==="UNSUPPORTED" && hasNum) return "수정/삭제 권장";
    if(st==="SUPPORTED" && tags.length===0) return "확인 권장";
    if(st==="PARTIAL") return "수정 또는 보류 권장";
    return "검토 필요";
  }

  function reasonCodesOf(r){
    const reasons = [];
    const tags = riskTags(r).map(t=>t.key);
    const miss = (r.missing_numbers||[]);

    // Coverage↑ / Proximity↓
    if(tags.includes("DISPERSED")){
      reasons.push("R: 근거분산(coverage↑ proximity↓)");
    }
    if(tags.includes("NUM_MISSING") && miss.length>0){
      reasons.push(`R: 숫자누락 ${miss.length}개`);
    }
    if(tags.includes("DIR_CONFLICT")){
      reasons.push("R: 방향성 모순");
    }
    if(tags.includes("UNIT_MISMATCH")){
      reasons.push("R: 단위 불일치");
    }

    const st = r.overall_raw || "UNSUPPORTED";
    if(st==="UNSUPPORTED") reasons.push("R: 근거미확인");
    else if(st==="PARTIAL") reasons.push("R: 부분근거");

    return reasons;
  }

  // priority score (lower is higher priority)
  function priorityRank(r){
    const tags = riskTags(r).map(t=>t.key);
    const nums = hasNumbers(r);

    let base = 4;
    if(r.overall_raw==="UNSUPPORTED") base = nums ? 0 : 1;
    else if(r.overall_raw==="PARTIAL") base = nums ? 2 : 3;
    else base = 5; // supported last

    let penalty = 0;
    if(tags.includes("NUM_MISSING")) penalty -= 0.6;
    if(tags.includes("BIG_NUM_MISSING")) penalty -= 0.4;
    if(tags.includes("DIR_CONFLICT")) penalty -= 0.6;
    if(tags.includes("UNIT_MISMATCH")) penalty -= 0.2;
    if(tags.includes("DISPERSED")) penalty -= 0.15;

    // encourage low scores first
    const s = r.score_overall || 0;
    return base + penalty + (1.0 - Math.min(1.0, Math.max(0.0, s)));
  }

  // --- filters ---
  const q = el("q");
  const idFilter = el("idFilter");
  const overallFilter = el("overallFilter");
  const riskFilter = el("riskFilter");
  const decisionFilter = el("decisionFilter");
  const sortBy = el("sortBy");

  function parseIdFilter(s){
    s = (s||"").trim();
    if(!s) return null;
    // accept: 1-10, 12, 15-20
    const parts = s.split(",").map(x=>x.trim()).filter(Boolean);
    const ranges = [];
    for(const p of parts){
      const m = p.match(/^(\d+)\s*-\s*(\d+)$/);
      if(m){
        const a = parseInt(m[1],10), b = parseInt(m[2],10);
        ranges.push([Math.min(a,b), Math.max(a,b)]);
      }else if(/^\d+$/.test(p)){
        const v = parseInt(p,10);
        ranges.push([v,v]);
      }
    }
    if(!ranges.length) return null;
    return ranges;
  }
  function idInRanges(id, ranges){
    for(const [a,b] of ranges){
      if(id>=a && id<=b) return true;
    }
    return false;
  }
  function matchQuery(r, query){
    query = (query||"").trim().toLowerCase();
    if(!query) return true;
    return String(r.claim||"").toLowerCase().includes(query) ||
           (r.numbers||[]).join(" ").toLowerCase().includes(query);
  }

  function applyFilters(){
    const query = q.value;
    const ranges = parseIdFilter(idFilter.value);
    const overall = overallFilter.value;
    const risk = riskFilter.value;
    const decision = decisionFilter.value;

    const out = [];
    for(const r of rows){
      if(!matchQuery(r, query)) continue;
      if(ranges && !idInRanges(r.id, ranges)) continue;
      if(overall!=="ALL" && r.overall_raw!==overall) continue;

      const tags = riskTags(r).map(t=>t.key);
      if(risk!=="ALL" && !tags.includes(risk)) continue;

      const d = getDecision(r.id).decision || "NONE";
      if(decision==="NONE" && d!=="NONE") continue;
      if(decision!=="ALL" && decision!=="NONE" && d!==decision) continue;

      out.push(r);
    }

    // sort
    const s = sortBy.value;
    if(s==="priority"){
      out.sort((a,b)=>priorityRank(a)-priorityRank(b));
    }else if(s==="id_asc"){
      out.sort((a,b)=>a.id-b.id);
    }else if(s==="score_desc"){
      out.sort((a,b)=>(b.score_overall||0)-(a.score_overall||0));
    }else if(s==="score_asc"){
      out.sort((a,b)=>(a.score_overall||0)-(b.score_overall||0));
    }

    // count info
    el("countInfo").textContent = `표시 ${out.length} / 전체 ${rows.length}`;
    return out;
  }

  // --- render queue ---
  const tbody = document.querySelector("#queueTable tbody");
  let selectedId = null;

  function updateBatchActions(){
    const checked = document.querySelectorAll("#queueTable tbody input[type='checkbox']:checked");
    const count = checked.length;
    const batchActions = el("batchActions");
    const batchCount = el("batchCount");
    if(count > 0){
      batchActions.style.display = "flex";
      batchCount.textContent = `선택: ${count}건`;
    }else{
      batchActions.style.display = "none";
    }
  }

  function renderQueue(list){
    tbody.innerHTML = "";
    for(const r of list){
      const d = getDecision(r.id).decision || "NONE";
      const tags = riskTags(r);
      const tagsHtml = tags.slice(0,3).map(t=>`<span class="tag ${t.tone}">${t.label}</span>`).join(" ");
      const rec = recommendedActionOf(r);
      const claimShort = String(r.claim||"").replace(/\s+/g," ").slice(0,80);

      const tr = document.createElement("tr");
      tr.dataset.id = r.id;
      if(needsAttention(r, d)) tr.classList.add("needs-attn");
      if(selectedId===r.id) tr.classList.add("selected");
      if(d==="CONFIRMED") tr.classList.add("confirmed");
      else if(d==="EDIT") tr.classList.add("needs-edit");
      else if(d==="DELETE") tr.classList.add("deleted");
      else if(d==="HOLD") tr.classList.add("hold");
      
      // Create checkbox cell (separate column)
      const checkboxTd = document.createElement("td");
      checkboxTd.style.textAlign = "center";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.id = r.id;
      checkbox.addEventListener("change", updateBatchActions);
      checkbox.addEventListener("click", (e)=>e.stopPropagation());
      checkboxTd.appendChild(checkbox);
      
      // Create ID cell (ID number only)
      const idTd = document.createElement("td");
      idTd.className = "mono";
      idTd.textContent = r.id;
      
      // Create status cell
      const statusTd = document.createElement("td");
      statusTd.innerHTML = `
        <div class="small">자동판정</div>
        ${statusPill(r.overall_raw)}
        <div class="small" style="margin-top:8px">최종조치</div>
        ${decisionPill(d)}
      `;
      
      // Create score cell
      const scoreTd = document.createElement("td");
      scoreTd.innerHTML = scoreBar(r.score_overall||0);
      
      // Create claim cell
      const claimTd = document.createElement("td");
      const attnHtml = needsAttention(r, d)
        ? `<span class="attn" onclick="jumpToClaim(${r.id}); event.stopPropagation();">확인필요</span>`
        : "";
      claimTd.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px">${tagsHtml}</div>
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px">${attnHtml}<span class="small"><b>추천 조치</b>: ${rec}</span></div>
        <div style="color:#cbd5e1">${claimShort}${(String(r.claim||"").length>80)?"…":""}</div>
      `;
      
      tr.appendChild(checkboxTd);
      tr.appendChild(idTd);
      tr.appendChild(statusTd);
      tr.appendChild(scoreTd);
      tr.appendChild(claimTd);
      
      tr.addEventListener("click", (e)=>{
        if(e.target.type === "checkbox") return;
        selectedId = r.id;
        document.querySelectorAll("#queueTable tbody tr").forEach(x=>x.classList.remove("selected"));
        tr.classList.add("selected");
        renderDetail(r);
        // Scroll to detail panel
        const detailPanel = el("detailPanel");
        if(detailPanel){
          scrollToElementTop(detailPanel, 10);
        }
      });
      tbody.appendChild(tr);
    }
    updateBatchActions();
  }

  // Batch operations
  el("selectAllCheckbox").addEventListener("change", (e)=>{
    const checked = e.target.checked;
    document.querySelectorAll("#queueTable tbody input[type='checkbox']").forEach(cb=>{
      cb.checked = checked;
    });
    updateBatchActions();
  });

  el("batchSelectAllBtn").onclick = ()=>{
    document.querySelectorAll("#queueTable tbody input[type='checkbox']").forEach(cb=>{
      cb.checked = true;
    });
    el("selectAllCheckbox").checked = true;
    updateBatchActions();
  };

  el("batchDeselectAllBtn").onclick = ()=>{
    document.querySelectorAll("#queueTable tbody input[type='checkbox']").forEach(cb=>{
      cb.checked = false;
    });
    el("selectAllCheckbox").checked = false;
    updateBatchActions();
  };

  el("batchApplyBtn").onclick = ()=>{
    const decision = el("batchDecision").value;
    const checked = document.querySelectorAll("#queueTable tbody input[type='checkbox']:checked");
    if(checked.length === 0){
      alert("선택된 항목이 없습니다.");
      return;
    }
    if(!confirm(`선택한 ${checked.length}건을 "${decisionPill(decision).replace(/<[^>]*>/g, '')}"로 처리하시겠습니까?`)){
      return;
    }
    
    checked.forEach(cb=>{
      const id = parseInt(cb.dataset.id, 10);
      const d = getDecision(id);
      setDecision(id, decision, d.note || "");
    });
    
    // Uncheck all
    document.querySelectorAll("#queueTable tbody input[type='checkbox']").forEach(cb=>{
      cb.checked = false;
    });
    el("selectAllCheckbox").checked = false;
    
    // Refresh
    buildKpis();
    renderQueue(currentList);
    if(selectedId !== null){
      const found = currentList.find(r=>r.id===selectedId);
      if(found) renderDetail(found);
    }
  };

  // --- render detail ---
  function metricRow(label, value, hint){
    return `<div class="row"><div>${label}</div><div class="mono">${value}${hint?` <span>(${hint})</span>`:""}</div></div>`;
  }

  function evidenceHtml(kind, evs){
    if(!evs || !evs.length){
      return `<div class="evidence-card"><div class="small">근거가 없습니다.</div></div>`;
    }
    const top = evs[0];
    const meta = (kind==="PDF")
      ? `p.<b>${top.page}</b> / chunk <b>${top.chunk_id}</b> / score <b>${fmt(top.score)}</b>`
      : `row <b>${top.row}</b> / chunk <b>${top.chunk_id}</b> / score <b>${fmt(top.score)}</b>`;

    let more = "";
    if(evs.length>1){
      more += `<details style="margin-top:10px"><summary class="small">More evidence (${evs.length-1})</summary>`;
      for(const ev of evs.slice(1)){
        const m = (kind==="PDF")
          ? `p.<b>${ev.page}</b> / chunk <b>${ev.chunk_id}</b> / score <b>${fmt(ev.score)}</b>`
          : `row <b>${ev.row}</b> / chunk <b>${ev.chunk_id}</b> / score <b>${fmt(ev.score)}</b>`;
        more += `<div class="evidence-card" style="margin-top:10px">
                  <div class="e-head"><div class="meta">${m}</div></div>
                  <div class="snippet">${ev.snippet_html||""}</div>
                </div>`;
      }
      more += `</details>`;
    }

    return `
      <div class="evidence-card">
        <div class="e-head"><div class="meta">${meta}</div></div>
        <div class="snippet">${top.snippet_html||""}</div>
        ${more}
      </div>
    `;
  }

  function renderDetail(r){
    const stLabel = statusLabel(r.overall_raw);
    el("detailHint").textContent = `ID ${r.id} 선택됨 • 자동판정(통합): ${stLabel}`;
    const d = getDecision(r.id);
    const tags = riskTags(r);
    const tagsHtml = tags.map(t=>`<span class="tag ${t.tone}">${t.label}</span>`).join(" ");
    const rec = recommendedActionOf(r);
    const reasons = reasonCodesOf(r);
    const reasonsHtml = reasons.length
      ? `<div style="margin-top:10px"><div class="small"><b>판정 사유(Reason codes)</b></div><div class="small" style="margin-top:6px; white-space:pre-line">${reasons.join("\n")}</div></div>`
      : "";

    const nums = (r.numbers||[]);
    const missing = (r.missing_numbers||[]);
    const numBox = nums.length
      ? `<div class="small">숫자(${nums.length}): <span class="mono">${nums.join(" · ")}</span></div>
         <div class="small" style="margin-top:6px">누락(${missing.length}): ${missing.length?`<span class="mono">${missing.join(" · ")}</span>`:"-"}</div>`
      : `<div class="small">숫자 없음</div>`;

    const pdfMetrics = `
      ${metricRow("Final", fmt(r.score_pdf||0))}
      ${metricRow("sim", fmt(r.sim_pdf||0))}
      ${metricRow("coverage", fmt(r.cov_pdf||0))}
      ${metricRow("proximity", fmt(r.prox_pdf||0))}
      ${metricRow("big-num", fmt(r.num_pdf||0), `exact ${r.exact_big_pdf||0}`)}
      ${metricRow("unit", fmt(r.unit_pdf||0))}
      ${metricRow("direction", fmt(r.dir_pdf||1))}
    `;
    const xlsxMetrics = `
      ${metricRow("Final", fmt(r.score_xlsx||0))}
      ${metricRow("sim", fmt(r.sim_xlsx||0))}
      ${metricRow("coverage", fmt(r.cov_xlsx||0))}
      ${metricRow("proximity", fmt(r.prox_xlsx||0))}
      ${metricRow("big-num", fmt(r.num_xlsx||0), `exact ${r.exact_big_xlsx||0}`)}
      ${metricRow("unit", fmt(r.unit_xlsx||0))}
      ${metricRow("direction", fmt(r.dir_xlsx||1))}
    `;

    const pane = el("detailPane");
    pane.innerHTML = `
      <div class="claim-box">
        <div class="decision-select-box">
          <div class="small" style="margin-bottom:6px"><b>자동판정</b>: ${statusPill(r.overall_raw)}</div>
          <div class="small" style="margin-bottom:10px"><b>추천 조치</b>: ${rec}</div>
          <select id="decisionSelect">
            <option value="NONE">미결정</option>
            <option value="CONFIRMED">확인</option>
            <option value="EDIT">수정필요</option>
            <option value="DELETE">삭제</option>
            <option value="HOLD">보류(추가근거 필요)</option>
          </select>
        </div>
        <div class="claim-content">
          <div class="claim-title">Claim</div>
          <div class="claim-text">${r.claim||""}</div>
        </div>
      </div>

      <div class="evidence-tabs">
        <div class="tab active" id="tabPdf">PDF 근거</div>
        <div class="tab" id="tabXlsx">XLSX 근거</div>
      </div>

      <div id="evPdf">${evidenceHtml("PDF", r.pdf_evidence||[])}</div>
      <div id="evXlsx" style="display:none">${evidenceHtml("XLSX", r.xlsx_evidence||[])}</div>

      <div class="claim-box" style="margin-top:12px">
        <div style="display:flex; flex-wrap:wrap; gap:6px">${tagsHtml}</div>
        <div style="margin-top:10px">${numBox}</div>
        ${reasonsHtml}
      </div>


      <div class="grid2">
        <div class="card">
          <h3>PDF 평가 • ${statusPill(r.pdf_raw)}</h3>
          ${pdfMetrics}
          ${r.dir_conflict_pdf?`<div style="margin-top:8px"><span class="tag bad">방향성 모순 의심</span></div>`:""}
        </div>
        <div class="card">
          <h3>XLSX 평가 • ${statusPill(r.xlsx_raw)}</h3>
          ${xlsxMetrics}
          ${r.dir_conflict_xlsx?`<div style="margin-top:8px"><span class="tag bad">방향성 모순 의심</span></div>`:""}
        </div>
      </div>

      <div class="review-box">
        <label>메모 (공유/감사 목적 기록)</label>
        <textarea id="decisionNote" placeholder="메모: 수정 사유, 필요한 추가 근거, 코멘트 등"></textarea>
        <div class="review-actions">
          <button class="btn primary" id="saveDecisionBtn">저장</button>
          <button class="btn" id="copyClaimBtn">Claim 복사</button>
        </div>
        <div class="small" id="savedHint" style="margin-top:8px"></div>
      </div>
    `;

    const sel = el("decisionSelect");
    const note = el("decisionNote");
    const savedHint = el("savedHint");
    sel.value = d.decision || "NONE";
    note.value = d.note || "";
    if(d.updated_at){
      savedHint.textContent = `최근 저장: ${new Date(d.updated_at).toLocaleString()}`;
    }else{
      savedHint.textContent = "";
    }

    el("saveDecisionBtn").onclick = ()=>{
      setDecision(r.id, sel.value, note.value);
      savedHint.textContent = `저장됨: ${new Date().toLocaleString()}`;
      // rerender queue to show decision badge and color
      renderQueue(currentList);
      renderDocGradeBadge();
      buildFullAttnList();
    };
    // Auto-save on decision change
    sel.addEventListener("change", ()=>{
      setDecision(r.id, sel.value, note.value);
      savedHint.textContent = `자동 저장됨: ${new Date().toLocaleString()}`;
      renderQueue(currentList);
      renderDocGradeBadge();
      buildFullAttnList();
    });
    el("copyClaimBtn").onclick = async ()=>{
      try{
        await navigator.clipboard.writeText(r.claim||"");
        savedHint.textContent = "Claim 복사 완료";
      }catch(e){
        savedHint.textContent = "복사 실패(브라우저 권한 확인)";
      }
    };

    // tabs
    el("tabPdf").onclick = ()=>{
      el("tabPdf").classList.add("active");
      el("tabXlsx").classList.remove("active");
      el("evPdf").style.display = "";
      el("evXlsx").style.display = "none";
    };
    el("tabXlsx").onclick = ()=>{
      el("tabXlsx").classList.add("active");
      el("tabPdf").classList.remove("active");
      el("evPdf").style.display = "none";
      el("evXlsx").style.display = "";
    };
  }

  // --- KPI summary ---
  function buildKpis(list){
    const total = rows.length;
    const supported = rows.filter(r=>r.overall_raw==="SUPPORTED").length;
    const partial = rows.filter(r=>r.overall_raw==="PARTIAL").length;
    const unsupported = rows.filter(r=>r.overall_raw==="UNSUPPORTED").length;

    const tagsAll = rows.flatMap(r=>riskTags(r).map(t=>t.key));
    const count = (k)=>tagsAll.filter(x=>x===k).length;

    const decisions = loadDecisions();
    let decided = 0;
    for(const k in decisions){ if(decisions[k] && decisions[k].decision && decisions[k].decision!=="NONE") decided++; }

    const kpis = [
      {label:"총 Claim", value: total, hint:"검증 대상 문장 수"},
      {label:"근거확인", value: supported, hint:"자동 검증 통과"},
      {label:"부분확인", value: partial, hint:"추가 검토 필요"},
      {label:"근거미확인", value: unsupported, hint:"수정/삭제 우선"},
      {label:"숫자 리스크", value: count("NUM_MISSING"), hint:"숫자 누락 포함"},
      {label:"검토 완료", value: decided, hint:"로컬 저장 기준"},
    ];
    const wrap = el("kpis");
    wrap.innerHTML = "";
    for(const k of kpis){
      wrap.innerHTML += `
        <div class="kpi">
          <div class="label">${k.label}</div>
          <div class="value">${k.value}</div>
          <div class="hint">${k.hint}</div>
        </div>
      `;
    }
  }

  // --- export ---
  function downloadText(filename, text, mime){
    const blob = new Blob([text], {type: mime||"text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(()=>URL.revokeObjectURL(url), 2000);
  }
  function toCsv(rowsArr){
    const cols = [
      // sharing-friendly canonical columns
      "claim_id","claim_text","system_status",
      "quality_w","quality_a","quality_tags",
      "decision","note","updated_at",
      "pdf_score","xlsx_score",
      "missing_numbers","risk_tags",

      // keep legacy columns (backward compatibility)
      "id","overall","overall_raw","score_overall",
      "pdf","pdf_raw","score_pdf",
      "xlsx","xlsx_raw","score_xlsx",
      "num_total","num_missing","numbers",
      "claim"
    ];
    const dec = loadDecisions();
    const lines = [];
    lines.push(cols.join(","));
    for(const r of rowsArr){
      const d = dec[String(r.id)] || {decision:"NONE", note:"", updated_at:""};
      const tagsObj = riskTags(r);
      const tagsLabel = tagsObj.map(t=>t.label).join("|");
      const tagsKey = tagsObj.map(t=>t.key).join("|");
      const w = weightOf(r);
      const a = actionScoreOf(r, d.decision||"NONE", d.note||"");
      const obj = {
        claim_id: r.id,
        claim_text: (r.claim||"").replace(/\n/g," ").replace(/"/g,'""'),
        system_status: r.overall_raw || "",
        quality_w: pct1(w),
        quality_a: pct1(a),
        quality_tags: tagsKey,
        id:r.id,
        overall:r.overall, overall_raw:r.overall_raw, score_overall:fmt(r.score_overall||0),
        pdf:r.pdf, pdf_raw:r.pdf_raw, score_pdf:fmt(r.score_pdf||0),
        xlsx:r.xlsx, xlsx_raw:r.xlsx_raw, score_xlsx:fmt(r.score_xlsx||0),
        risk_tags:tagsLabel,
        decision:d.decision||"NONE",
        note:(d.note||"").replace(/\n/g," "),
        updated_at:d.updated_at||"",
        pdf_score: fmt(r.score_pdf||0),
        xlsx_score: fmt(r.score_xlsx||0),
        num_total:r.num_total||0,
        num_missing:r.num_missing||0,
        numbers:(r.numbers||[]).join("|"),
        missing_numbers:(r.missing_numbers||[]).join("|"),
        claim:(r.claim||"").replace(/\n/g," ").replace(/"/g,'""'),
      };
      const line = cols.map(c=>{
        const v = (obj[c]!==undefined && obj[c]!==null) ? String(obj[c]) : "";
        return `"${v.replace(/"/g,'""')}"`;
      }).join(",");
      lines.push(line);
    }
    return lines.join("\n");
  }

  el("exportDecisionsBtn").onclick = ()=>{
    const csv = toCsv(rows); // full
    downloadText("review_decisions.csv", csv, "text/csv;charset=utf-8");
  };
  el("exportViewBtn").onclick = ()=>{
    const csv = toCsv(currentList);
    downloadText("current_view.csv", csv, "text/csv;charset=utf-8");
  };
  el("printBtn").onclick = ()=> window.print();
  el("clearDecisionsBtn").onclick = ()=>{
    if(confirm("저장된 검토결과(로컬)를 모두 초기화할까요?")){
      clearDecisions();
      buildKpis();
      renderQueue(currentList);
      const pane = el("detailPane");
      pane.innerHTML = "";
      el("detailHint").textContent = "좌측에서 항목을 선택하세요";
      const out = el("reviewScoreOut");
      if(out) out.textContent = "";
    }
  };

  // --- review scoring ---
  function clamp01(x){ return Math.max(0, Math.min(1, x)); }
  function pct1(x){ return Math.round(x*10)/10; } // 1 decimal

  function hasNumberInClaim(r){
    // 가능한 필드명들을 폭넓게 지원
    if(r && r.has_number === true) return true;
    const arr1 = r && r.numbers_in_claim;
    if(Array.isArray(arr1) && arr1.length > 0) return true;
    const arr2 = r && r.numbers;
    if(Array.isArray(arr2) && arr2.length > 0) return true;
    if((r && (r.num_total||0)) > 0) return true;
    return false;
  }

  // claim 중요도 가중치 w_i
  function weightOf(r){
    let w = 1.0;
    const st = r.overall_raw || "UNSUPPORTED";
    const tags = riskTags(r).map(t=>t.key);

    // 숫자 포함
    if(hasNumberInClaim(r)) w += 0.3;

    // 리스크 태그
    if(tags.includes("NUM_MISSING")) w += 0.4;
    if(tags.includes("BIG_NUM_MISSING")) w += 0.6;
    if(tags.includes("DIR_CONFLICT")) w += 0.7;
    if(tags.includes("UNIT_MISMATCH")) w += 0.3;

    // 자동 status
    if(st==="UNSUPPORTED") w += 0.4;
    else if(st==="PARTIAL") w += 0.2;
    return w;
  }

  // claim 단위 조치 점수 a_i (0~1)
  // (요청 정의) 문서 품질 기여도 Q_i
  // - CONFIRMED: 1.0 - penalty(auto_status, risk_tags), clamp 0~1
  // - EDIT: 0.0
  // - HOLD: 0.5
  // - NONE: 0.6
  // - DELETE: 0.2 (감점)
  function actionScoreOf(r, decision, note){
    // v8.5 (micro-tuned): raise post-review score while keeping EDIT/DELETE penalties meaningful.
    // - CONFIRMED base/boost slightly higher
    // - Risk-tag penalties much lighter when CONFIRMED, evidence score can offset them
    // - EDIT/DELETE penalties depend smoothly on evidence score
    decision = (decision || "NONE").toUpperCase();

    const stRaw = (r.overall_raw || "UNSUPPORTED").toUpperCase();
    const s = clamp01(r.score_overall || 0);

    // Adjust auto status by evidence score
    let st = (stRaw === "SUPPORTED" || stRaw === "PARTIAL") ? stRaw : "UNSUPPORTED";
    if(st === "SUPPORTED" && s < 0.35) st = "PARTIAL";
    if(st === "PARTIAL" && s < 0.20) st = "UNSUPPORTED";

    // 1) Base by human decision
    const B = {
      "CONFIRMED": 0.92,
      "EDIT": 0.70,
      "DELETE": 0.52,
      "HOLD": 0.58,
      "NONE": 0.25,
    };
    let q = (B[decision] != null) ? B[decision] : B["NONE"];

    // 2) Auto x decision (gentle)
    const A = {
      "SUPPORTED":   { "CONFIRMED": +0.05, "EDIT": -0.05, "DELETE": -0.14, "HOLD": +0.00, "NONE": -0.08 },
      "PARTIAL":     { "CONFIRMED": +0.04, "EDIT": -0.08, "DELETE": -0.18, "HOLD": -0.02, "NONE": -0.12 },
      "UNSUPPORTED": { "CONFIRMED": +0.03, "EDIT": -0.10, "DELETE": -0.22, "HOLD": -0.05, "NONE": -0.20 },
    };
    q += (A[st] && A[st][decision] != null) ? A[st][decision] : 0.0;

    // 3) Evidence score boost / penalty
    if(decision === "CONFIRMED"){
      const k = (st === "SUPPORTED") ? 0.13 : (st === "PARTIAL" ? 0.11 : 0.10);
      q += k * s;
      if(s >= 0.85) q += 0.015;
    }else if(decision === "EDIT"){
      q += 0.08 * s;
      q -= 0.05 * (1 - s);
    }else if(decision === "DELETE"){
      q += 0.06 * s;
      q -= 0.14 * (1 - s);
    }else if(decision === "HOLD"){
      q += 0.05 * s;
    }else{
      q += 0.02 * s;
    }

    // 4) Risk-tag penalty
    const tags = riskTags(r).map(t=>t.key);
    let penalty = 0.0;

    const isConfirmed = (decision === "CONFIRMED");

    const confScale = isConfirmed ? (0.12 + 0.20 * (1 - s)) : 1.0; // s=1 -> 0.12, s=0 -> 0.32
    const missScale = isConfirmed ? (0.10 + 0.25 * (1 - s)) : (0.55 + 0.45 * (1 - s));

    if(tags.includes("DIR_CONFLICT")) penalty += 0.22 * confScale;
    if(tags.includes("NUM_CONFLICT")) penalty += 0.22 * confScale;

    if(tags.includes("BIG_NUM_MISSING")) penalty += 0.12 * missScale;
    if(tags.includes("UNIT_MISMATCH"))  penalty += 0.08 * missScale;
    if(tags.includes("NUM_MISSING"))    penalty += 0.05 * missScale;

    q = q - penalty;

    if(isConfirmed) q = Math.max(q, 0.78);

    return clamp01(q);
  }

  // auto-human mismatch (가중치 %) 계산용
  // - mismatch = "자동판정 관점에서 기대되는 조치"와 다른 결정을 내린 경우(미결정 제외)
  function alignmentScoreOf(r, decision){
    decision = decision || "NONE";
    if(decision === "NONE") return null; // exclude from mismatch rate (미검토)

    const st = r.overall_raw || "UNSUPPORTED";

    // heuristics (직관적 기준)
    // - SUPPORTED: CONFIRMED가 정합, 나머지는 불일치
    // - PARTIAL: EDIT/HOLD가 정합, CONFIRMED/DELETE는 불일치
    // - UNSUPPORTED: EDIT/DELETE/HOLD가 정합, CONFIRMED는 불일치
    let ok = true;
    if(st === "SUPPORTED"){
      ok = (decision === "CONFIRMED");
    }else if(st === "PARTIAL"){
      ok = (decision === "EDIT" || decision === "HOLD");
    }else{
      ok = (decision === "EDIT" || decision === "DELETE" || decision === "HOLD");
    }
    return ok ? 0.0 : 1.0; // 1.0 = mismatch
  }

  function computeReviewScores(){
    const dec = loadDecisions();

    const total = rows.length || 0;
    let reviewed = 0; // decision != NONE
    let deleteCount = 0;

    // quality score: include DELETE (it harmed draft quality even if reviewer removed it)
    let weightSum = 0.0;
    let qualitySum = 0.0; // Σ(W_i * Q_i)

    // weighted ratios
    let editWeightSum = 0.0;
    let mismatchWeightSum = 0.0;

    // audit risk count
    let auditRisk = 0;
    let requiredUndecided = 0;

    for(const r of rows){
      const d = dec[String(r.id)] || {decision:"NONE", note:""};
      const decision = d.decision || "NONE";
      if(decision !== "NONE") reviewed += 1;
      if(decision === "NONE" && isEvidenceInsufficient(r)) requiredUndecided += 1;
      if(decision === "DELETE") deleteCount += 1;

      const w = weightOf(r);

      const qi = actionScoreOf(r, decision, d.note || "");
      weightSum += w;
      qualitySum += w * qi;

      if(decision === "EDIT") editWeightSum += w;

      const mis = alignmentScoreOf(r, decision);
      if(mis !== null) mismatchWeightSum += w * mis;

      const st = r.overall_raw || "UNSUPPORTED";
      if(st === "UNSUPPORTED" && decision === "CONFIRMED" && hasNumberInClaim(r)){
        auditRisk += 1;
      }
    }

    const qualityScore = weightSum ? (100.0 * qualitySum / weightSum) : 0.0;
    const completion = total ? (100.0 * reviewed / total) : 0.0;

    const editShareWeighted = weightSum ? (100.0 * editWeightSum / weightSum) : 0.0;
    const mismatchShareWeighted = weightSum ? (100.0 * mismatchWeightSum / weightSum) : 0.0;
    const deleteShare = total ? (100.0 * deleteCount / total) : 0.0;

    return {
      total,
      reviewed,
      weightSum,
      qualityScore,
      completion,
      auditRisk,
      editShareWeighted,
      mismatchShareWeighted,
      deleteShare,
      requiredUndecided,
    };
  }

  function gradeFromScore(score){
    // score: 0~100
    if(score >= 95) return "S";
    if(score >= 90) return "A+";
    if(score >= 85) return "A";
    if(score >= 80) return "A-";
    if(score >= 75) return "B+";
    if(score >= 70) return "B";
    if(score >= 65) return "B-";
    if(score >= 60) return "C+";
    if(score >= 55) return "C";
    if(score >= 50) return "C-";
    return "F";
  }

  function renderDocGradeBadge(){
    const badge = el("docGradeBadge");
    if(!badge) return;
    const s = computeReviewScores();
    const score = s.qualityScore || 0;
    const grade = gradeFromScore(score);
    const letterEl = badge.querySelector(".grade-letter");
    const scoreEl = badge.querySelector(".grade-score");
    if(letterEl) letterEl.textContent = grade;
    if(scoreEl) scoreEl.textContent = `${pct1(score)}/100`;

    // subtle tone by grade (no hard colors required, but add class for readability)
    badge.classList.remove("gS","gA","gB","gC","gF");
    if(grade === "S" || grade.startsWith("A")) badge.classList.add("gA");
    else if(grade.startsWith("B")) badge.classList.add("gB");
    else if(grade.startsWith("C")) badge.classList.add("gC");
    else badge.classList.add("gF");
  }


  function filterByReviewKpiType(type){
    const dec = loadDecisions();
    const ids = [];

    for(const r of rows){
      const d = dec[String(r.id)] || {decision:"NONE", note:""};
      const decision = d.decision || "NONE";

      if(type === "quality_kept"){
        if(decision !== "DELETE") ids.push(r.id);
      }else if(type === "completion_unreviewed"){
        if(decision === "NONE") ids.push(r.id);
      }else if(type === "audit_risk"){
        const st = r.overall_raw || "UNSUPPORTED";
        if(st==="UNSUPPORTED" && decision==="CONFIRMED" && hasNumberInClaim(r)) ids.push(r.id);
      }else if(type === "edit"){
        if(decision === "EDIT") ids.push(r.id);
      }else if(type === "mismatch"){
        const mis = alignmentScoreOf(r, decision);
        if(mis === 1.0) ids.push(r.id);
      }else if(type === "delete"){
        if(decision === "DELETE") ids.push(r.id);
      }
    }

    if(ids.length === 0){
      alert("해당 항목이 없습니다.");
      return;
    }
    el("idFilter").value = ids.join(",");
    refresh();
  }

  function showReviewKpiPopup(type){
    const s = computeReviewScores();
    const dec = loadDecisions();

    let title = "";
    let currentValue = "";
    let meaning = "";
    let formula = "";
    let stats = "";
    let detail = "";
    let filterType = "";

    if(type === "quality"){
      title = "문서품질점수(0~100)";
      currentValue = `${pct1(s.qualityScore)}/100`;
      meaning = "각 claim의 품질기여도(Q_i)를 리스크 가중치(W_i)로 가중평균한 점수입니다. (DELETE도 감점으로 반영)";
      formula = "Score = 100 × Σ(W_i × Q_i) / Σ(W_i)";
      stats = `kept 가중치 합 ΣW = ${pct1(s.keptWeightSum)} | 전체 ${s.total}건`;
      detail = [
        "Q_i(품질기여도):",
        "- CONFIRMED: 1 - penalty(auto_status, risk_tags) (0~1 clamp)",
        "- EDIT: 0.0 (품질저해)",
        "- HOLD: 0.5 (미확정)",
        "- NONE: 0.6 (미검토)",
        "- DELETE: 0.2 (감점)",
      ].join("\n");
      filterType = "quality_kept";
    }else if(type === "completion"){
      title = "검토완료도(%)";
      currentValue = `${Math.round(s.completion)}%`;
      meaning = "전체 claim 중 reviewer가 최종조치(미결정 제외)를 내린 비율입니다.";
      formula = "Completion = (결정 건수 / 전체 건수) × 100";
      const undecidedN = rows.filter(r=> (dec[String(r.id)]||{decision:'NONE'}).decision==="NONE").length;
      stats = `전체 ${s.total}건 | 미결정 ${undecidedN}건`;
      detail = "완료도가 낮으면 문서 품질이 아직 확정되지 않았다는 의미이므로, 우선적으로 미결정 항목을 처리하세요.";
      filterType = "completion_unreviewed";
    }else if(type === "audit"){
      title = "감사리스크(건)";
      currentValue = `${s.auditRisk}건`;
      meaning = "자동판정이 UNSUPPORTED(근거미확인)인데도 숫자가 포함된 claim을 CONFIRMED(확인)으로 처리한 건수입니다.";
      formula = "COUNT(status==UNSUPPORTED AND decision==CONFIRMED AND has_number)";
      const ids = rows.filter(r=>{
        const d = dec[String(r.id)] || {decision:"NONE"};
        const st = r.overall_raw || "UNSUPPORTED";
        return st==="UNSUPPORTED" && d.decision==="CONFIRMED" && hasNumberInClaim(r);
      }).map(r=>r.id);
      stats = `${s.auditRisk}건 (ID: ${ids.slice(0,50).join(", ")}${ids.length>50?"…":""})`;
      detail = "감사 관점에서 가장 문제가 될 수 있는 케이스입니다. 원문 근거를 재확인하고 필요 시 수정/삭제 조치를 권장합니다.";
      filterType = "audit_risk";
    }else if(type === "editShare"){
      title = "수정필요비중(가중치 %)";
      currentValue = `${Math.round(s.editShareWeighted)}%`;
      meaning = "kept 대상(DELETE 제외) 중 EDIT로 처리된 항목의 중요도(가중치) 비중입니다.";
      formula = "Σ(W_i for decision==EDIT) / Σ(W_i for decision!=DELETE) × 100";
      stats = `kept ΣW=${pct1(s.keptWeightSum)} 기준`;
      detail = "EDIT는 품질저해로 간주되어 Q_i=0.0으로 반영됩니다. 리스크 큰 문장(가중치↑)이 EDIT로 많을수록 점수가 크게 하락합니다.";
      filterType = "edit";
    }else if(type === "mismatch"){
      title = "자동-인간 불일치율(가중치 %)";
      currentValue = `${Math.round(s.mismatchShareWeighted)}%`;
      meaning = "자동판정(system) 기준으로 기대되는 조치와 reviewer의 최종조치가 어긋난 비율(가중치 기준)입니다. 미결정(NONE)은 제외합니다.";
      formula = "Σ(W_i × mismatch_i) / Σ(W_i for decision!=DELETE) × 100,  mismatch_i∈{0,1}";
      stats = `kept ΣW=${pct1(s.keptWeightSum)} 기준`;
      detail = [
        "정합(불일치=0) 기준:",
        "- SUPPORTED → CONFIRMED",
        "- PARTIAL → EDIT/HOLD",
        "- UNSUPPORTED → EDIT/DELETE/HOLD",
        "(그 외는 불일치=1로 계산)",
      ].join("\n");
      filterType = "mismatch";
    }else if(type === "delete"){
      title = "삭제비중(%)";
      currentValue = `${Math.round(s.deleteShare)}%`;
      meaning = "전체 claim 중 DELETE로 처리된 비율입니다. DELETE는 문서 품질점수 산정에서 제외됩니다.";
      formula = "(DELETE 건수 / 전체 건수) × 100";
      const delN = rows.filter(r=> (dec[String(r.id)]||{decision:'NONE'}).decision==="DELETE").length;
      stats = `DELETE ${delN}건 / 전체 ${s.total}건`;
      detail = "DELETE는 '문서에 남지 않는' 조치로 가정하여, 품질점수 계산 대상(kept)에서 제외합니다.";
      filterType = "delete";
    }

    const modal = document.createElement("div");
    modal.className = "modal-overlay";
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <div class="modal-title">${title}</div>
          <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
        </div>
        <div class="modal-body">
          <div><strong>현재 값:</strong> ${currentValue}</div>
          <h4>의미</h4>
          <div>${meaning}</div>
          <h4>계산식</h4>
          <div style="white-space:pre-line"><code>${formula}</code></div>
          <h4>실제 데이터</h4>
          <div>${stats}</div>
          <div style="margin-top:12px;">
            <button class="btn primary" onclick="filterByReviewKpiType('${filterType}'); this.closest('.modal-overlay').remove();">해당 항목만 보기</button>
          </div>
          <h4>상세 설명</h4>
          <div style="white-space: pre-line; color:#94a3b8; font-size:12px;">${detail}</div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener("click", (e)=>{ if(e.target === modal) modal.remove(); });
  }

  el("reviewScoreBtn").onclick = ()=>{
    const s = computeReviewScores();
    const line1 = [
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('quality')">
문서품질점수: ${pct1(s.qualityScore)}/100</span>`,
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('completion')">검토완료도: ${Math.round(s.completion)}%</span>`,
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('audit')">감사리스크: ${s.auditRisk}건</span>`,
    ].join(" | ");
    const line2 = [
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('editShare')">수정필요비중(가중): ${Math.round(s.editShareWeighted)}%</span>`,
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('mismatch')">자동-인간 불일치율(가중): ${Math.round(s.mismatchShareWeighted)}%</span>`,
      `<span style="cursor:pointer; text-decoration:underline; color:#93c5fd;" onclick="showReviewKpiPopup('delete')">삭제비중: ${Math.round(s.deleteShare)}%</span>`,
    ].join(" | ");
    const out = el("reviewScoreOut");
    if(out) out.innerHTML = `${line1}<br/>${line2}`;
  };
  window.showReviewKpiPopup = showReviewKpiPopup;
  window.filterByReviewKpiType = filterByReviewKpiType;

  // --- main loop ---
  let currentList = [];
  function refresh(){
    currentList = applyFilters();
    buildKpis(currentList);
    renderDocGradeBadge();
    renderQueue(currentList);
    // keep selection if still present
    if(selectedId!==null){
      const found = currentList.find(r=>r.id===selectedId);
      if(found) renderDetail(found);
    }
  }

  ["input","change"].forEach(evt=>{
    q.addEventListener(evt, refresh);
    idFilter.addEventListener(evt, refresh);
    overallFilter.addEventListener(evt, refresh);
    riskFilter.addEventListener(evt, refresh);
    decisionFilter.addEventListener(evt, refresh);
    sortBy.addEventListener(evt, refresh);
  });

  // sheet tab handlers (default: full)
  el("sheetTabFull").onclick = ()=> showSheet("full");
  el("sheetTabReport").onclick = ()=> showSheet("report");
  showSheet("full");

  refresh();
  buildFullAttnList();
})();
""")
    html.append("</script></div></body></html>")

    # Write HTML file
    html_content = "".join(html)
    html_path.write_text(html_content, encoding="utf-8")

    out: Dict[str, str] = {
        "json": str(report_json_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "total": str(len(results)),
    }
    if xlsx_path:
        out["xlsx"] = str(xlsx_path)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Markdown result file path (.md/.txt)")
    ap.add_argument("--pdf", required=True, help="Source PDF path (text selectable)")
    ap.add_argument("--xlsx", nargs="*", default=[], help="XLSX source file path(s), 5~6 files (e.g. --xlsx a.xlsx b.xlsx)")
    ap.add_argument("--out_dir", default="reports", help="Output directory")
    ap.add_argument("--top_k", type=int, default=3, help="Top K evidence chunks to keep")

    ap.add_argument("--th_num_supported", type=float, default=0.30)
    ap.add_argument("--th_num_partial", type=float, default=0.22)
    ap.add_argument("--th_text_supported", type=float, default=0.30)
    ap.add_argument("--th_text_partial", type=float, default=0.22)

    args = ap.parse_args()

    md_path = Path(args.md)
    pdf_path = Path(args.pdf)
    xlsx_paths = [Path(p) for p in args.xlsx] if args.xlsx else []
    out_dir = Path(args.out_dir)

    if not md_path.exists():
        raise FileNotFoundError(f"MD file not found: {md_path}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    out = verify(
        md_path,
        pdf_path,
        out_dir,
        top_k=args.top_k,
        xlsx_paths=xlsx_paths if xlsx_paths else None,
        th_num_supported=args.th_num_supported,
        th_num_partial=args.th_num_partial,
        th_text_supported=args.th_text_supported,
        th_text_partial=args.th_text_partial,
    )

    print("Done.")
    print(f"HTML: {out['html']}")
    print(f"CSV : {out['csv']}")
    print(f"JSON: {out['json']}")
    if "xlsx" in out:
        print(f"XLSX: {out['xlsx']}")


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        main()
    else:
        run_gui()