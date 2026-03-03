from dataclasses import dataclass


@dataclass
class Evidence:
    page: int
    chunk_id: int
    score: float
    snippet: str


@dataclass
class ClaimResult:
    claim_id: int
    claim_text: str

    status_overall: str  # SUPPORTED / PARTIAL / UNSUPPORTED
    status_pdf: str
    status_xlsx: str

    best_score: float
    best_score_pdf: float
    best_score_xlsx: float

    sim_best_pdf: float
    sim_best_xlsx: float

    coverage_pdf: float
    coverage_xlsx: float

    proximity_pdf: float
    proximity_xlsx: float

    big_num_recall_pdf: float
    big_num_recall_xlsx: float

    exact_big_num_matches_pdf: int
    exact_big_num_matches_xlsx: int

    unit_match_pdf: float
    unit_match_xlsx: float

    direction_score_pdf: float
    direction_score_xlsx: float

    direction_conflict_pdf: bool
    direction_conflict_xlsx: bool

    numbers_in_claim: list
    matched_numbers: list
    matched_numbers_pdf: list
    matched_numbers_xlsx: list

    missing_numbers_pdf: list
    missing_numbers_xlsx: list
    missing_numbers_all: list

    num_conflict_pdf: bool
    num_conflict_xlsx: bool

    evidence: list  # PDF evidence
    evidence_xlsx: list  # XLSX evidence (multiple files)
