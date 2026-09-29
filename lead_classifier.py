"""
lead_classifier.py  (v2 — Industrial Grade)
============================================
AI-powered Lead Scoring & Classification System with:
  - Intent Gate : semantic query analysis that conditionally modulates all other scores
  - Conditional Weights : cold query dampens budget/urgency/DM; hot query gets full weight
  - Manual Weight Overrides : loaded from config.json (no code changes needed)
  - Richer Semantic Scoring : separate embeddings + expanded anchor bank
  - Ensemble-ready : feature extractor also used by the ML pipeline in lead model.ipynb

Architecture
------------
  [Query + Message]
        |
  Intent Gate  (sigmoid of hot_sim - cold_sim)  → gate ∈ [0,1]
        |
        ├── gate < 0.35 (COLD signal) → dampen budget, urgency, company_size; zero DM score
        ├── gate 0.35–0.60 (NEUTRAL)  → apply normal weights
        └── gate > 0.60 (HOT signal)  → full weights × manual multipliers
        |
  Structured Scorer (max 50)    ←  budget · urgency · DM · company_size · lead_source
  Semantic Scorer   (max 50)    ←  intent · clarity · service_scope
        |
  Final Score (0–100) → HOT (≥70) | WARM (40–69) | COLD (<40)

Usage
-----
  from lead_classifier import load_lead, classify_lead, print_report

  lead   = load_lead("data.json")
  result = classify_lead(lead)
  print_report(result)
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration (used if config.json is missing)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict = {
    "weights": {
        "budget": 1.0, "urgency": 1.0, "decision_maker": 1.2,
        "company_size": 0.8, "lead_source": 0.7,
        "intent": 1.3, "clarity": 1.0, "service_scope": 0.9,
    },
    "intent_gate": {
        "hot_threshold": 0.60, "cold_threshold": 0.35,
        "cold_dampening": True, "min_dampen_factor": 0.08,
    },
    "thresholds": {"hot": 70, "warm": 40},
    "semantic": {
        "intent_max_pts": 20, "clarity_max_pts": 20,
        "scope_max_pts": 10, "sigmoid_k": 8.0,
    },
    "budget_sanity": {
        # Min budget (Lakhs) considered realistic per service-count tier.
        # If actual budget < threshold AND query sounds hot (gate > hot_threshold),
        # the system detects a contradiction (possible prank / unrealistic lead):
        #   - semantic scores dampened by contradiction_dampen multiplier
        #   - final score hard-capped at hard_cap_score (Warm zone)
        "enabled":              True,
        "min_for_1_service_l":  0.50,   # min 50K for single service
        "min_for_2_service_l":  1.00,   # min 1L for 2 services
        "min_for_3_service_l":  2.00,   # min 2L for 3 services
        "min_for_4plus_l":      4.00,   # min 4L for 4+ services
        "contradiction_dampen": 0.25,   # semantic score multiplier when prank detected
        "hard_cap_score":       52.0,   # absolute max score when prank detected
    },
}


_CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config(path: str | Path | None = None) -> dict:
    """Load config from config.json.  Falls back to built-in defaults."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Strip _comment keys
            def strip(d):
                if isinstance(d, dict):
                    return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
                return d
            return strip(raw)
        except Exception:
            pass
    return _DEFAULT_CONFIG.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """Per-dimension scores produced by the scoring pipeline."""

    # Intent gate (0–1): how strong the query intent signal is
    intent_gate: float = 0.5

    # Structured dimensions (before weights; max is config-defined)
    budget_score:         float = 0.0
    urgency_score:        float = 0.0
    decision_maker_score: float = 0.0
    company_size_score:   float = 0.0
    lead_source_score:    float = 0.0

    # Semantic dimensions (after weights)
    intent_score:         float = 0.0
    clarity_score:        float = 0.0
    service_scope_score:  float = 0.0

    # Totals and classification
    structured_total: float = 0.0
    semantic_total:   float = 0.0
    final_score:      float = 0.0
    tier:             str   = ""

    # Gate analysis
    gate_hi_sim:  float = 0.0  # similarity to high-intent anchors
    gate_lo_sim:  float = 0.0  # similarity to low-intent anchors
    dampen_factor: float = 1.0  # multiplier applied to structured scores

    # Budget sanity check results
    budget_sanity_triggered: bool  = False  # True if prank/contradiction detected
    budget_sanity_dampen:    float = 1.0    # multiplier applied to semantic scores
    budget_sanity_cap:       float = 100.0  # hard cap on final score

    # Hard COLD overrides (bypass all scoring and force Cold tier)
    ni_override_triggered:    bool  = False  # True if "not interested" hard-kill fired
    prank_budget_triggered:   bool  = False  # True if budget is absurdly low (prank floor)
    log_budget_per_duration:  float = 0.0   # log(budget_INR) / duration_months feature

    # Human-readable explanations
    rationale: list[str] = field(default_factory=list)



@dataclass
class LeadResult:
    lead:      dict[str, Any]
    breakdown: ScoreBreakdown


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_lead(path: str | Path) -> dict[str, Any]:
    """Load a single lead from a JSON file. If the file is a list, returns the first element."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data[0] if isinstance(data, list) else data


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load all leads from a JSON file (array or single object)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def load_leads_from_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load leads from a CSV file with pipe-separated multi-value fields."""
    import csv
    leads = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                if isinstance(v, str) and "|" in v:
                    row[k] = [x.strip() for x in v.split("|")]
                elif isinstance(v, str) and v.lower() == "true":
                    row[k] = True
                elif isinstance(v, str) and v.lower() == "false":
                    row[k] = False
            leads.append(dict(row))
    return leads


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_budget_inr(budget_str: str) -> float:
    """
    Extract upper bound from INR budget string. Returns value in Lakhs.
    E.g. '8,00,000 - 12,00,000' -> 12.0.  Returns 0.0 on failure.
    """
    if not budget_str:
        return 0.0
    cleaned = re.sub(r"[\u20b9Rs.,\s]", "", str(budget_str))
    nums = re.findall(r"\d+", cleaned)
    if not nums:
        return 0.0
    return max(int(n) for n in nums) / 100_000


def parse_company_size(size_str: str) -> int:
    """Extract upper bound from company size string. E.g. '51-200' -> 200."""
    nums = re.findall(r"\d+", str(size_str))
    return max(int(n) for n in nums) if nums else 0


def parse_duration_months(duration_str: str) -> float:
    """
    Extract the upper bound in months from a project_duration string.
    Handles patterns like '4-6 months', '12+ months', 'Ongoing (12+ months)',
    '2-3 weeks' (converted → fraction of month), 'Unknown', etc.
    Returns 0.0 when no numeric information is found.
    """
    if not duration_str:
        return 0.0
    s = str(duration_str).lower()
    # Weeks → convert to months
    week_nums = re.findall(r"(\d+)\s*(?:-\s*\d+)?\s*week", s)
    if week_nums:
        return max(int(w) for w in week_nums) / 4.33
    # Months (pick the largest number found)
    month_nums = re.findall(r"\d+", s)
    if month_nums:
        return float(max(int(n) for n in month_nums))
    return 0.0


def compute_log_budget_per_duration(lead: dict) -> float:
    """
    Compute  log(budget_INR) / duration_months  as a composite quality signal.

    Interpretation
    --------------
    High value → substantial budget relative to duration → credible lead.
    Very low value (tiny budget OR very long duration) → financially implausible.
    Zero  → missing/undecipherable budget or duration.

    Returns the raw float; callers decide how to use it for scoring.
    """
    budget_lakhs   = parse_budget_inr(str(lead.get("estimated_budget", "")))
    duration_months = parse_duration_months(str(lead.get("project_duration", "")))

    if budget_lakhs <= 0 or duration_months <= 0:
        return 0.0

    budget_inr = budget_lakhs * 100_000            # convert lakhs → INR
    return np.log(budget_inr) / duration_months    # log10 to keep numbers sane




def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def mean_sim(query_vec: np.ndarray, anchor_vecs: np.ndarray) -> float:
    """Average cosine similarity of query_vec against a bank of anchor vectors."""
    return float(np.mean([cosine_sim(query_vec, av) for av in anchor_vecs])) if len(anchor_vecs) else 0.0


def max_sim(query_vec: np.ndarray, anchor_vecs: np.ndarray) -> float:
    """Maximum cosine similarity of query_vec against a bank of anchor vectors."""
    return float(np.max([cosine_sim(query_vec, av) for av in anchor_vecs])) if len(anchor_vecs) else 0.0


def _sigmoid(x: float, k: float = 8.0) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-k*x))."""
    kx = k * x
    if kx > 100:  return 1.0
    if kx < -100: return 0.0
    return 1.0 / (1.0 + np.exp(-kx))


# ──────────────────────────────────────────────────────────────────────────────
# Anchor bank (extended for richer semantic analysis)
# ──────────────────────────────────────────────────────────────────────────────

# HIGH buying-intent anchors
_HIGH_INTENT_ANCHORS = [
    "We are ready to start the project immediately and have the budget approved.",
    "Looking for an experienced team to build our product as soon as possible.",
    "Please share your portfolio. We would like to schedule a call this week.",
    "We want to launch within the next month and are serious about this investment.",
    "Budget is finalized and the board has approved. We need a committed partner.",
    "We have secured funding and are ready to onboard a development team.",
    "We cannot afford delays. Please share your availability for a kickoff call.",
    "This is a business-critical project. We are evaluating your proposal seriously.",
]

# LOW buying-intent / exploratory anchors
_LOW_INTENT_ANCHORS = [
    "Just browsing and curious about pricing.",
    "We are exploring options and have no timeline or budget yet.",
    "Not sure if we will proceed, just gathering information.",
    "We might consider this project sometime next year.",
    "Just wanted to know market rates, no plans to start soon.",
    "Budget is very limited, looking for the cheapest option.",
    "We are at a very early idea stage, nothing is decided.",
    "I am a student looking for an affordable developer.",
]

# URGENCY anchors — high urgency
_URGENCY_HIGH_ANCHORS = [
    "We need to start immediately. The project is time-sensitive.",
    "There is a hard launch deadline we cannot miss.",
    "We are under pressure to deliver this by end of quarter.",
    "The business is dependent on this — we cannot wait.",
]

# COMMITMENT anchors — signals of financial and operational commitment
_COMMITMENT_ANCHORS = [
    "We have already signed an NDA with our previous vendor.",
    "Our team is ready to work full-time with your developers.",
    "We will provide a dedicated product manager for this project.",
    "We have a signed purchase order and are ready to pay an advance.",
]

# CLEAR project scope anchors
_CLEAR_PROJECT_ANCHORS = [
    "The project requires specific features including user login, payment integration, and real-time notifications.",
    "We need both a mobile application and a web admin panel with clearly defined modules.",
    "The scope includes backend API, database design, and third-party service integrations.",
    "We have a detailed product specification document outlining all functional requirements.",
    "We need a scalable microservices backend with CI/CD pipeline and 99.9% uptime SLA.",
]

# VAGUE / under-defined project anchors
_VAGUE_PROJECT_ANCHORS = [
    "We want an app but are not sure what features to include.",
    "Something like that popular app but for our use case.",
    "We will figure out the requirements as we go.",
    "We just need something simple for now.",
    "Not sure about the technical details yet.",
]

# NOT INTERESTED — active disengagement / rejection anchors
# These are STRONGER than low-intent: the lead is explicitly opting out,
# not merely being exploratory. High similarity here should sharply reduce the gate.
_NOT_INTERESTED_ANCHORS = [
    "Not interested.",
    "I am not interested.",
    "We are not interested in this service.",
    "We are not interested in working with you.",
    "We don't need this service at the moment.",
    "We have decided not to proceed with this project.",
    "We already have an in-house team handling this, so we won't be moving forward.",
    "We are not looking for any development work right now.",
    "We have found another vendor and have already signed a contract with them.",
    "Our management has decided against this initiative for now.",
    "We have put this project on hold indefinitely.",
    "We are not interested in working with you at this time.",
    "This is not a priority for us and we have no plans to revisit it.",
    "We have changed our mind and will not be pursuing this project.",
    "Please remove us from your list. We are not interested.",
    "We do not require your services. Thank you.",
    "No requirement at all, please cancel.",
    "Stop contacting us, we are not interested.",
    "Not looking to hire anyone, we decided against it.",
    "Do not proceed, we chose another vendor.",
    "We don't think your services are promising or suitable for us.",
    "Your services do not look promising for our requirements.",
    "We are skeptical about your quality and not convinced you can deliver.",
    "We doubt your capabilities and do not think your team is a good fit.",
    "We do not have confidence in your services and will not proceed.",
    "We are not convinced by your proposal and will look elsewhere.",
]


def is_not_interested_text(text: str) -> tuple[bool, str]:
    """
    Fast, comprehensive lexical inspection for explicit disinterest / rejection / cancellation.
    Returns (is_ni: bool, reason: str).
    """
    if not text:
        return False, ""
    t = text.lower().strip()

    # Exact or near-exact short phrases
    if t in (
        "not interested", "not interested.", "not interested!",
        "no interest", "not interested at all", "no thanks",
        "don't need", "do not need", "no need", "cancel"
    ):
        return True, "explicit 'not interested' query"

    patterns = [
        (r"\bnot\s+interested\b", "contains 'not interested'"),
        (r"\b(don'?t|do\s+not|doesn'?t|does\s+not)\s+need\b", "stated does not need service"),
        (r"\b(don'?t|do\s+not|doesn'?t|does\s+not)\s+require\b", "stated does not require service"),
        (r"\b(don'?t|do\s+not|doesn'?t|does\s+not)\s+want\b", "stated does not want service"),
        (r"\bnot\s+looking\s+(for|to)\b", "stated not looking for service"),
        (r"\bno\s+longer\s+(interested|need|needed|require|required|looking|proceeding)\b", "no longer interested/required"),
        (r"\bdecided\s+(against|not\s+to)\b", "decided against proceeding"),
        (r"\b(found|chose|chosen|hired|went\s+with|go\s+with|going\s+with)\s+another\b", "selected another provider/vendor"),
        (r"\banother\s+(provider|vendor|agency|company|team|developer|partner)\b", "selected another provider/vendor"),
        (r"\b(have|got|using|handled\s+by)\s+(an\s+)?in-?house\b", "handling in-house"),
        (r"\b(put\s+on\s+hold|on\s+hold\s+indefinitely|project\s+on\s+hold|postponed\s+indefinitely)\b", "project on hold indefinitely"),
        (r"\bchanged\s+(our|my)\s+mind\b", "changed mind about project"),
        (r"\b(not|won'?t\s+be|will\s+not\s+be)\s+(going\s+forward|proceeding|pursuing|moving\s+forward)\b", "not moving forward"),
        (r"\b(remove\s+(us|me)|unsubscribe|stop\s+(contacting|emailing|calling|messaging)|do\s+not\s+contact|don'?t\s+contact)\b", "opt-out / stop contact requested"),
        (r"\b(no|zero)\s+requirement\b", "no requirement"),
        (r"\bnot\s+a\s+priority\b", "not a business priority"),
        (r"\bcancel\s+(this|my|our)?\s*(enquiry|inquiry|request|lead|project)?\b", "cancellation requested"),
        (r"\bclose\s+(this|my|our)?\s*(enquiry|inquiry|request)\b", "close request"),
        (r"\bdo\s+not\s+proceed\b", "do not proceed instruction"),
        (r"\b(not|un)\s*promising\b", "stated services not promising"),
        (r"\bdon'?t\s+think.*promising\b", "perception that services are unpromising"),
        (r"\bdon'?t\s+think\s+(your|the)\s+services\b", "negative evaluation of services"),
        (r"\b(skeptical|doubtful)\s+(about|of)\b", "expressed skepticism about services"),
        (r"\bnot\s+(convinced|impressed|confident)\b", "not convinced about services"),
        (r"\bnot\s+a\s+good\s+fit\b", "not a good fit"),
    ]
    for pat, reason in patterns:
        if re.search(pat, t):
            return True, reason

    return False, ""



# ──────────────────────────────────────────────────────────────────────────────
# Encoder singleton (loaded once, cached)
# ──────────────────────────────────────────────────────────────────────────────

_encoder_cache: Any = None

def _get_encoder():
    global _encoder_cache
    if _encoder_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder_cache = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    return _encoder_cache


def _encode(encoder, texts: list[str]) -> np.ndarray:
    return encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ──────────────────────────────────────────────────────────────────────────────
# Intent Gate
# ──────────────────────────────────────────────────────────────────────────────

def _query_keyword_net(text: str) -> float:
    """
    Fast lexical pre-screen of intent from the query field.
    Returns a net score in approximately [-1.5, +1]:
      +ve  = hot-keyword-rich  (approved, immediately, ready, urgent …)
      -ve  = cold-keyword-rich (curious, exploring, pricing, cheap …)
      −2×  = not-interested    (don't need, decided against, found vendor …)
             ↑ Active rejection counts DOUBLE vs passive coldness.

    Used alongside semantic similarity for a dual-signal gate.
    """
    t = text.lower()
    hot_hits = sum([
        "approved" in t, "immediately" in t, "this month" in t,
        "this week" in t, "start now" in t, "ready to start" in t,
        "kickoff" in t, "urgent" in t, "funded" in t,
        "investor" in t, "board" in t, "signed" in t,
        "serious" in t, "cannot afford delay" in t, "committed" in t,
    ])
    cold_hits = sum([
        "curious" in t, "exploring" in t, "just checking" in t,
        "how much" in t, "price" in t, "cost" in t,
        "very limited" in t, "cheap" in t, "affordable" in t,
        "no timeline" in t, "not sure" in t, "idea stage" in t,
        "someday" in t, "future" in t, "maybe" in t,
    ])
    # NOT INTERESTED — active rejection, counted as 2 cold hits each
    ni_hits = sum([
        "don't need" in t, "do not need" in t, "not interested" in t,
        "not looking" in t, "decided against" in t, "found another" in t,
        "in-house" in t, "put on hold" in t, "on hold" in t,
        "changed our mind" in t, "no longer" in t,
        "not going forward" in t, "not proceeding" in t,
        "remove us" in t, "unsubscribe" in t, "won't be" in t,
        "will not be" in t, "not thinking" in t,
    ])
    total = hot_hits + cold_hits + ni_hits
    return (hot_hits - cold_hits - 2 * ni_hits) / max(total + 1, 1)



def compute_intent_gate(lead: dict[str, Any], encoder, cfg: dict) -> tuple[float, dict]:
    """
    Analyze the QUERY field (primary) and MESSAGE field (secondary context)
    separately, then blend to produce a gate value ∈ [0, 1].

    WHY SEPARATE?
    When query + message are merged, a short vague query gets swamped by a
    long detailed message, making the gate blind to the real first signal.
    The `query` field is what the lead typed first — it is the most honest
    indicator of intent. Message is follow-up elaboration and gets less weight.

    Gate = sigmoid(query_weight × query_net + (1-query_weight) × message_net)

    Gate Interpretation
    -------------------
    < 0.35  → COLD signal  : dampen all other scores
    0.35–0.60 → NEUTRAL   : apply normal weights
    > 0.60  → HOT signal  : full weights applied

    Returns
    -------
    gate : float  ∈ [0, 1]
    vecs : dict   Pre-computed anchor vectors + per-field vectors for reuse.
    """
    # Config
    ig_cfg       = cfg.get("intent_gate", {})
    q_weight     = ig_cfg.get("query_weight", 0.75)          # query dominance [0-1]
    kw_blend     = ig_cfg.get("keyword_blend", 0.20)         # keyword net contribution
    ni_weight    = ig_cfg.get("not_interested_weight", 0.60) # NI sim subtraction factor
    k            = cfg.get("semantic", {}).get("sigmoid_k", 8.0)

    q_text = str(lead.get("query",   "")).strip()
    m_text = str(lead.get("message", "")).strip()

    # Pre-encode anchor banks (reused in semantic_scorer)
    hi_vecs  = _encode(encoder, _HIGH_INTENT_ANCHORS)
    lo_vecs  = _encode(encoder, _LOW_INTENT_ANCHORS)
    urg_vecs = _encode(encoder, _URGENCY_HIGH_ANCHORS)
    com_vecs = _encode(encoder, _COMMITMENT_ANCHORS)
    clr_vecs = _encode(encoder, _CLEAR_PROJECT_ANCHORS)
    vag_vecs = _encode(encoder, _VAGUE_PROJECT_ANCHORS)
    ni_vecs  = _encode(encoder, _NOT_INTERESTED_ANCHORS)   # active rejection

    # Direct lexical check for explicit rejection / not interested
    q_is_ni, q_ni_reason = is_not_interested_text(q_text)
    m_is_ni, m_ni_reason = is_not_interested_text(m_text)

    # ── Query signal (PRIMARY) ─────────────────────────────────────────────────
    if q_text:
        q_vec    = _encode(encoder, [q_text])[0]
        q_hi     = mean_sim(q_vec, hi_vecs)
        q_lo     = mean_sim(q_vec, lo_vecs)
        q_urg    = mean_sim(q_vec, urg_vecs)
        q_ni     = mean_sim(q_vec, ni_vecs)                 # not-interested mean sim
        q_ni_max = max_sim(q_vec, ni_vecs)                  # not-interested max sim
        q_kw     = _query_keyword_net(q_text)               # lexical boost (NI=2×)
        # q_ni subtracts directly — active rejection overrides hot signals
        q_net    = (q_hi - q_lo) + 0.25 * q_urg + kw_blend * q_kw - ni_weight * q_ni
    else:
        q_vec    = np.zeros(384, dtype=np.float32)
        q_hi = q_lo = q_urg = q_ni = q_ni_max = 0.0
        q_net    = 0.0
        q_kw     = 0.0

    # ── Message signal (SECONDARY — context, not primary intent indicator) ─────
    if m_text:
        m_vec    = _encode(encoder, [m_text])[0]
        m_hi     = mean_sim(m_vec, hi_vecs)
        m_lo     = mean_sim(m_vec, lo_vecs)
        m_urg    = mean_sim(m_vec, urg_vecs)
        m_ni     = mean_sim(m_vec, ni_vecs)                 # NI in message too
        m_ni_max = max_sim(m_vec, ni_vecs)
        m_net    = (m_hi - m_lo) + 0.15 * m_urg - 0.40 * m_ni
    else:
        m_vec    = np.zeros(384, dtype=np.float32)
        m_hi = m_lo = m_urg = m_ni = m_ni_max = 0.0
        m_net    = 0.0

    # If explicitly not interested, clamp gate to 0.0 immediately
    if q_is_ni or m_is_ni or q_ni_max >= 0.48 or m_ni_max >= 0.48:
        gate = 0.0
    else:
        # Blend: query dominates
        combined_net = q_weight * q_net + (1.0 - q_weight) * m_net
        gate         = round(_sigmoid(combined_net, k), 4)

    vecs = {
        "hi_vecs": hi_vecs, "lo_vecs": lo_vecs,
        "urg_vecs": urg_vecs, "com_vecs": com_vecs,
        "clr_vecs": clr_vecs, "vag_vecs": vag_vecs,
        "ni_vecs":  ni_vecs,
        # Per-field vectors (reused in semantic_scorer)
        "q_vec":  q_vec,  "q_hi": q_hi, "q_lo": q_lo, "q_ni": q_ni, "q_ni_max": q_ni_max,
        "m_vec":  m_vec,  "m_hi": m_hi, "m_lo": m_lo, "m_ni": m_ni, "m_ni_max": m_ni_max,
        "q_is_ni": q_is_ni, "q_ni_reason": q_ni_reason,
        "m_is_ni": m_is_ni, "m_ni_reason": m_ni_reason,
        # Legacy keys kept for compatibility
        "query_vec": q_vec, "hi_sim": q_hi, "lo_sim": q_lo,
        # Keyword signal
        "q_keyword_net": q_kw,
    }
    return gate, vecs




# ──────────────────────────────────────────────────────────────────────────────
# Structured Scorer — with conditional gate weighting
# ──────────────────────────────────────────────────────────────────────────────

def structured_scorer(lead: dict[str, Any], intent_gate: float,
                      cfg: dict) -> tuple[float, ScoreBreakdown]:
    """
    Rule-based scoring on structured fields, modulated by the intent gate.

    Conditional logic
    -----------------
    - intent_gate < cold_threshold :
        All score dimensions dampened by (gate / cold_threshold).
        Decision-maker score is zeroed (if they're not interested, DM status is irrelevant).
    - intent_gate >= hot_threshold :
        Full weights applied. Small additional urgency boost.
    - Otherwise (neutral zone) :
        Normal weights applied without gate modulation.

    Returns
    -------
    (weighted_total, ScoreBreakdown)
    """
    bd  = ScoreBreakdown()
    bd.intent_gate = intent_gate

    weights    = cfg.get("weights", _DEFAULT_CONFIG["weights"])
    ig_cfg     = cfg.get("intent_gate", _DEFAULT_CONFIG["intent_gate"])
    cold_thr   = ig_cfg.get("cold_threshold", 0.35)
    hot_thr    = ig_cfg.get("hot_threshold",  0.60)
    do_dampen  = ig_cfg.get("cold_dampening", True)
    min_dampen = ig_cfg.get("min_dampen_factor", 0.08)

    # ── Compute dampening factor ──────────────────────────────────────────────
    if do_dampen and intent_gate < cold_thr:
        dampen = max(intent_gate / cold_thr, min_dampen)
    else:
        dampen = 1.0
    bd.dampen_factor = dampen

    # ── Budget (raw max 15) ───────────────────────────────────────────────────
    budget_lakhs = parse_budget_inr(str(lead.get("estimated_budget", "")))
    if   budget_lakhs >= 10: raw_budget = 15.0
    elif budget_lakhs >= 5:  raw_budget = 11.0
    elif budget_lakhs >= 2:  raw_budget =  7.0
    elif budget_lakhs >  0:  raw_budget =  3.0
    else:                    raw_budget =  0.0
    bd.budget_score = round(raw_budget * weights.get("budget", 1.0) * dampen, 2)

    # ── Urgency (raw max 12) ──────────────────────────────────────────────────
    urgency    = str(lead.get("urgency", "")).strip().lower()
    start_date = str(lead.get("expected_start_date", "")).lower()
    urg_map    = {"high": 12.0, "medium": 7.0, "low": 3.0}
    raw_urg    = urg_map.get(urgency, 0.0)
    if "within 1 month" in start_date or "immediate" in start_date:
        raw_urg = min(raw_urg + 2.0, 12.0)
    bd.urgency_score = round(raw_urg * weights.get("urgency", 1.0) * dampen, 2)

    # ── Decision Maker (raw max 10) ───────────────────────────────────────────
    is_dm  = lead.get("decision_maker", False)
    raw_dm = 10.0 if is_dm else 0.0
    # CONDITIONAL: if cold signal is very strong, DM status is meaningless
    dm_dampen = 0.0 if (do_dampen and intent_gate < cold_thr * 0.7) else dampen
    bd.decision_maker_score = round(raw_dm * weights.get("decision_maker", 1.0) * dm_dampen, 2)

    # ── Company Size (raw max 8) ──────────────────────────────────────────────
    upper_size = parse_company_size(lead.get("company_size", ""))
    if   upper_size >= 200: raw_size = 8.0
    elif upper_size >= 50:  raw_size = 6.0
    elif upper_size >= 10:  raw_size = 4.0
    elif upper_size >  0:   raw_size = 2.0
    else:                   raw_size = 1.0
    bd.company_size_score = round(raw_size * weights.get("company_size", 0.8) * dampen, 2)

    # ── Lead Source (raw max 5) — NOT dampened (always informative) ───────────
    source = str(lead.get("lead_source", "")).lower()
    if   "website" in source or "referral" in source or "inbound" in source: raw_src = 5.0
    elif "linkedin" in source or "social" in source or "ad" in source:        raw_src = 3.0
    elif "cold" in source or "outbound" in source:                             raw_src = 1.0
    else:                                                                      raw_src = 2.0
    bd.lead_source_score = round(raw_src * weights.get("lead_source", 0.7), 2)

    bd.structured_total = round(
        bd.budget_score + bd.urgency_score + bd.decision_maker_score
        + bd.company_size_score + bd.lead_source_score, 2
    )
    return bd.structured_total, bd


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Scorer — richer anchor bank, separate text fields
# ──────────────────────────────────────────────────────────────────────────────

def semantic_scorer(lead: dict[str, Any], bd: ScoreBreakdown,
                    encoder, vecs: dict, cfg: dict) -> float:
    """
    Embedding-based scoring on three text fields separately.

    Populates bd.intent_score, bd.clarity_score, bd.service_scope_score, bd.rationale.
    Returns semantic_total in [0, 50] (after weight application).

    Scoring logic
    -------------
    Intent (query+message, max intent_max_pts):
        net_intent = (hi_sim - lo_sim + urg_sim + com_sim) normalized → [0,1] → scaled
    Clarity (project_description, max clarity_max_pts):
        net_clarity = (clr_sim - vag_sim) normalized → [0,1] → scaled
    Service Scope (service_required list, max scope_max_pts):
        Tiered by count: 0→0, 1→3, 2→5, 3→8, 4+→10
    """
    sem_cfg = cfg.get("semantic", _DEFAULT_CONFIG["semantic"])
    weights = cfg.get("weights",  _DEFAULT_CONFIG["weights"])

    intent_max = sem_cfg.get("intent_max_pts",  20)
    clarity_max= sem_cfg.get("clarity_max_pts", 20)
    scope_max  = sem_cfg.get("scope_max_pts",   10)
    k          = sem_cfg.get("sigmoid_k",        8.0)

    hi_vecs  = vecs.get("hi_vecs",  np.zeros((1, 384)))
    lo_vecs  = vecs.get("lo_vecs",  np.zeros((1, 384)))
    urg_vecs = vecs.get("urg_vecs", np.zeros((1, 384)))
    com_vecs = vecs.get("com_vecs", np.zeros((1, 384)))
    clr_vecs = vecs.get("clr_vecs", np.zeros((1, 384)))
    vag_vecs = vecs.get("vag_vecs", np.zeros((1, 384)))

    # ── Intent Score (query + message) ───────────────────────────────────────
    qvec = vecs.get("query_vec")
    hi_s = vecs.get("hi_sim", 0.0)
    lo_s = vecs.get("lo_sim", 0.0)

    if qvec is not None:
        urg_s = mean_sim(qvec, urg_vecs)
        com_s = mean_sim(qvec, com_vecs)
        # Weighted net: intent dominates, urgency and commitment add bonus
        net_int  = (hi_s - lo_s) + 0.25 * urg_s + 0.15 * com_s
        gate_int = _sigmoid(net_int, k)
        bd.intent_score = round(gate_int * intent_max * weights.get("intent", 1.3), 2)
        bd.gate_hi_sim  = hi_s
        bd.gate_lo_sim  = lo_s

        if hi_s > 0.42:
            bd.rationale.append("[STRONG] Query shows strong buying intent and urgency signals")
        elif hi_s > 0.28:
            bd.rationale.append("[MODERATE] Query shows moderate interest with some intent signals")
        else:
            bd.rationale.append("[WEAK] Query is vague or shows low commitment")

        if lo_s > 0.38:
            bd.rationale.append("[ALERT] Message contains exploratory or low-commitment language")
        if urg_s > 0.35:
            bd.rationale.append("[BONUS] Urgency signals detected in query — timeline pressure noted")
        if com_s > 0.32:
            bd.rationale.append("[BONUS] Commitment signals detected — budget/team readiness mentioned")
    else:
        bd.intent_score = round(5.0 * weights.get("intent", 1.3), 2)
        bd.rationale.append("[MISSING] No query/message text provided for semantic analysis")

    # Cap intent score
    bd.intent_score = min(bd.intent_score, intent_max)

    # ── Clarity Score (project_description) ──────────────────────────────────
    desc_text = str(lead.get("project_description", "")).strip()
    if desc_text:
        dvec    = _encode(encoder, [desc_text])[0]
        clr_s   = mean_sim(dvec, clr_vecs)
        vag_s   = mean_sim(dvec, vag_vecs)
        net_clr = (clr_s - vag_s + 1.0) / 2.0   # rescale to [0,1]
        bd.clarity_score = round(net_clr * clarity_max * weights.get("clarity", 1.0), 2)

        if clr_s > 0.42:
            bd.rationale.append("[STRONG] Project description is detailed and well-scoped")
        elif clr_s > 0.27:
            bd.rationale.append("[MODERATE] Project description has some specificity, but more detail needed")
        else:
            bd.rationale.append("[WEAK] Project description is vague — discovery call recommended")
    else:
        bd.clarity_score = round(2.0 * weights.get("clarity", 1.0), 2)
        bd.rationale.append("[MISSING] No project description provided")

    bd.clarity_score = min(bd.clarity_score, clarity_max)

    # ── Service Scope Score ───────────────────────────────────────────────────
    services = lead.get("service_required", [])
    if isinstance(services, str):
        services = [s.strip() for s in services.split("|") if s.strip()]
    n = len(services)
    if   n >= 4: raw_scope = scope_max
    elif n == 3: raw_scope = 8.0
    elif n == 2: raw_scope = 5.0
    elif n == 1: raw_scope = 3.0
    else:        raw_scope = 0.0
    bd.service_scope_score = round(raw_scope * weights.get("service_scope", 0.9), 2)
    bd.service_scope_score = min(bd.service_scope_score, scope_max)

    svc_desc = f"{n} service(s): {', '.join(services)}" if services else "none specified"
    bd.rationale.append(
        f"[SCOPE] {('Large' if n>=4 else 'Good' if n==3 else 'Moderate' if n==2 else 'Narrow' if n==1 else 'No')} "
        f"scope — {svc_desc}"
    )

    bd.semantic_total = round(
        bd.intent_score + bd.clarity_score + bd.service_scope_score, 2
    )
    return bd.semantic_total


# ──────────────────────────────────────────────────────────────────────────────
# Tier assignment
# ──────────────────────────────────────────────────────────────────────────────

def assign_tier(score: float, cfg: dict) -> str:
    thr = cfg.get("thresholds", _DEFAULT_CONFIG["thresholds"])
    if score >= thr.get("hot", 70):  return "Hot"
    if score >= thr.get("warm", 40): return "Warm"
    return "Cold"


# ──────────────────────────────────────────────────────────────────────────────
# Budget Sanity Gate  (prank / contradiction detection)
# ──────────────────────────────────────────────────────────────────────────────

def budget_sanity_check(lead: dict[str, Any], bd: ScoreBreakdown,
                        intent_gate: float, cfg: dict) -> None:
    """
    Detects contradictions between query intent and declared budget.

    The problem this solves
    -----------------------
    A prank lead writes a very hot-sounding query ("budget approved, start
    immediately, need a full SaaS platform") but declares a budget of Rs.5,000.
    Without this check the high intent gate leaves semantic scores untouched,
    pushing the final score into HOT even though the budget is absurd.

    Logic
    -----
    1. Compute minimum realistic budget for this project's service count.
    2. If actual_budget < min_budget AND intent_gate >= hot_threshold:
         → Contradiction detected (hot language + micro budget).
         → Dampen semantic scores by contradiction_dampen multiplier.
         → Hard-cap final score at hard_cap_score (Warm zone, never Hot).
         → Insert [!!! PRANK / CONTRADICTION DETECTED !!!] in rationale.

    Modifies bd in-place.  Does NOT touch structured scores (those are already
    budget-aware).  Only semantic scores are dampened since those are text-driven
    and therefore spoofable with a convincing query.
    """
    bs_cfg = cfg.get("budget_sanity", _DEFAULT_CONFIG.get("budget_sanity", {}))
    if not bs_cfg.get("enabled", True):
        return

    hot_thr  = cfg.get("intent_gate", {}).get("hot_threshold", 0.60)
    dampen   = bs_cfg.get("contradiction_dampen", 0.25)
    hard_cap = bs_cfg.get("hard_cap_score", 52.0)

    # Minimum realistic budget per service count
    services = lead.get("service_required", [])
    if isinstance(services, str):
        services = [s.strip() for s in services.split("|") if s.strip()]
    n_svc = len(services)

    if   n_svc >= 4: min_budget_l = bs_cfg.get("min_for_4plus_l",     4.00)
    elif n_svc == 3: min_budget_l = bs_cfg.get("min_for_3_service_l", 2.00)
    elif n_svc == 2: min_budget_l = bs_cfg.get("min_for_2_service_l", 1.00)
    elif n_svc == 1: min_budget_l = bs_cfg.get("min_for_1_service_l", 0.50)
    else:            min_budget_l = 0.0   # no services → no budget constraint

    actual_l = parse_budget_inr(str(lead.get("estimated_budget", "")))

    is_contradiction = (
        min_budget_l > 0
        and actual_l < min_budget_l
        and intent_gate >= hot_thr
    )

    if is_contradiction:
        bd.budget_sanity_triggered = True
        bd.budget_sanity_dampen    = dampen
        bd.budget_sanity_cap       = hard_cap

        bd.intent_score        = round(bd.intent_score        * dampen, 2)
        bd.clarity_score       = round(bd.clarity_score       * dampen, 2)
        bd.service_scope_score = round(bd.service_scope_score * dampen, 2)
        bd.semantic_total      = round(
            bd.intent_score + bd.clarity_score + bd.service_scope_score, 2)

        bd.rationale.insert(0,
            f"[!!! PRANK / CONTRADICTION DETECTED !!!] "
            f"Intent gate={intent_gate:.2f} (HOT query) but budget={actual_l:.2f}L "
            f"is below minimum {min_budget_l:.1f}L for {n_svc} service(s). "
            f"Semantic scores dampened {dampen:.0%}. Score hard-capped at {hard_cap}."
        )
    else:
        bd.budget_sanity_triggered = False
        bd.budget_sanity_dampen    = 1.0
        bd.budget_sanity_cap       = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Absolute COLD Overrides  (hard-kill rules — bypass all scoring)
# ──────────────────────────────────────────────────────────────────────────────

def absolute_cold_overrides(lead: dict[str, Any], bd: ScoreBreakdown,
                            gate: float, vecs: dict, cfg: dict) -> bool:
    """
    Deterministic hard-kill rules that force COLD tier regardless of any
    computed score.  Returns True if a hard-kill was triggered.

    Rule 1 — Not Interested Override
    ---------------------------------
    Triggered when ANY of the following is true:
      a) Intent gate < ni_hard_gate (default 0.15) — extremely cold gate score
      b) The not-interested semantic similarity in the query exceeds
         ni_sim_threshold (default 0.55) — very high NI anchor match
      c) The keyword net already caught NI phrasing (q_keyword_net very negative)

    Rule 2 — Prank / Micro Budget Override
    ----------------------------------------
    Triggered when estimated_budget is provided but the absolute INR value
    is ≤ prank_budget_max_inr (default ₹5,000), regardless of query intent.
    A budget of ₹1,000 is clearly not serious — skip scoring entirely.

    When triggered the function:
      - Zeros all scores in bd
      - Sets final_score = 0.0 and tier = "Cold"
      - Inserts a prominent rationale entry
      - Sets the corresponding flag (ni_override_triggered / prank_budget_triggered)
    """
    ov_cfg           = cfg.get("absolute_cold_overrides", {})
    ni_hard_gate     = ov_cfg.get("ni_hard_gate",         0.25)
    ni_sim_threshold = ov_cfg.get("ni_sim_threshold",     0.35)
    prank_max_inr    = ov_cfg.get("prank_budget_max_inr", 5_000)

    # ── Rule 1: Not Interested ────────────────────────────────────────────────
    q_text = str(lead.get("query", "")).strip()
    m_text = str(lead.get("message", "")).strip()

    q_is_ni   = vecs.get("q_is_ni", False) or is_not_interested_text(q_text)[0]
    m_is_ni   = vecs.get("m_is_ni", False) or is_not_interested_text(m_text)[0]
    q_ni_max  = vecs.get("q_ni_max", 0.0)
    m_ni_max  = vecs.get("m_ni_max", 0.0)
    q_ni      = vecs.get("q_ni", 0.0)      # semantic NI mean similarity for query
    m_ni      = vecs.get("m_ni", 0.0)      # semantic NI mean similarity for message
    q_kw_net  = vecs.get("q_keyword_net", 0.0)

    ni_triggered = (
        q_is_ni                                          # explicit query disengagement / rejection
        or m_is_ni                                      # explicit message disengagement / rejection
        or q_ni_max >= 0.45                              # query very close to a specific NI anchor
        or m_ni_max >= 0.45                              # message very close to a specific NI anchor
        or q_ni  >= ni_sim_threshold                     # query high average NI similarity
        or m_ni  >= ni_sim_threshold                     # message high average NI similarity
        or gate  <= ni_hard_gate                         # gate is near zero / strongly cold
        or q_kw_net <= -0.30                             # dominant NI keyword signal
    )

    if ni_triggered:
        bd.ni_override_triggered = True
        _zero_all_scores(bd)
        reason_detail = (
            vecs.get("q_ni_reason")
            or is_not_interested_text(q_text)[1]
            or vecs.get("m_ni_reason")
            or is_not_interested_text(m_text)[1]
            or f"gate={gate:.2f}, q_ni_max={q_ni_max:.2f}, q_ni_sim={q_ni:.2f}"
        )
        bd.rationale.insert(0,
            f"[HARD KILL -- NOT INTERESTED] Lead explicitly indicated disinterest ({reason_detail}). "
            f"Directly classified as COLD lead."
        )
        return True

    # ── Rule 2: Prank / Micro Budget ─────────────────────────────────────────
    budget_str   = str(lead.get("estimated_budget", "")).strip()
    budget_lakhs = parse_budget_inr(budget_str)
    budget_inr   = budget_lakhs * 100_000

    svcs = lead.get("service_required", [])
    if isinstance(svcs, str): svcs = [s.strip() for s in svcs.split("|") if s.strip()]
    n_svc = len(svcs)
    bs = cfg.get("budget_sanity", {})
    min_bl = bs.get(f"min_for_{min(n_svc, 3)}_service_l", bs.get("min_for_4plus_l", 4.0)) if n_svc >= 4 else bs.get(f"min_for_{n_svc}_service_l", 0.0)

    # Only trigger if a budget was actually provided (skip leads with no budget)
    has_budget = bool(budget_str and budget_str.lower() not in
                      ("", "not decided", "unknown", "n/a", "tbd"))

    is_prank = has_budget and (
        (0 < budget_inr <= prank_max_inr) or
        (min_bl > 0 and budget_lakhs < min_bl and (budget_lakhs < 0.50 or budget_inr < 50000) and n_svc >= 2)
    )

    if is_prank:
        bd.prank_budget_triggered = True
        _zero_all_scores(bd)
        bd.rationale.insert(0,
            f"[HARD KILL -- PRANK BUDGET] Declared budget of ₹{budget_inr:,.0f} for {n_svc} services fails "
            f"minimum realistic economic threshold (min req: ₹{min_bl*100000:,.0f}). "
            f"This is not a viable lead. Classified as COLD."
        )
        return True

    return False


def _zero_all_scores(bd: ScoreBreakdown) -> None:
    """Zero every scoring field in bd and set final_score=0 / tier='Cold'."""
    bd.budget_score         = 0.0
    bd.urgency_score        = 0.0
    bd.decision_maker_score = 0.0
    bd.company_size_score   = 0.0
    bd.lead_source_score    = 0.0
    bd.structured_total     = 0.0
    bd.intent_score         = 0.0
    bd.clarity_score        = 0.0
    bd.service_scope_score  = 0.0
    bd.semantic_total       = 0.0
    bd.budget_sanity_cap    = 0.0
    bd.final_score          = 0.0
    bd.tier                 = "Cold"


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def classify_lead(lead: dict[str, Any],
                  config: dict | None = None,
                  encoder=None) -> LeadResult:
    """
    Full pipeline:
      1. Intent gate              — semantic analysis of query+message
      2. Structured scorer        — budget, urgency, DM, size, source (gate-modulated)
                                    + log(budget)/duration penalty applied here
      3. Semantic scorer          — intent, clarity, service scope (embedding-based)
      3.5 Budget sanity gate      — prank/contradiction detection and score capping
      3.75 Absolute COLD overrides — hard-kill for NI leads & micro-budget pranks
      4. Combine & classify       — final score 0-100, tier Hot/Warm/Cold

    Parameters
    ----------
    lead   : dict   Lead data dictionary.
    config : dict   Optional override config. If None, loads from config.json.
    encoder         Optional pre-loaded SentenceTransformer. If None, loads & caches.

    Returns
    -------
    LeadResult containing lead dict and a complete ScoreBreakdown.
    """
    cfg = config if config is not None else load_config()
    enc = encoder if encoder is not None else _get_encoder()

    # Step 1: intent gate
    gate, vecs = compute_intent_gate(lead, enc, cfg)

    # Step 2: structured scoring with conditional gate weighting
    struct_total, bd = structured_scorer(lead, gate, cfg)

    # Step 2.5: compute log(budget)/duration feature and apply penalty
    lbd              = compute_log_budget_per_duration(lead)
    bd.log_budget_per_duration = round(lbd, 4)
    lbd_cfg          = cfg.get("log_budget_duration", {})
    lbd_penalty_thr  = lbd_cfg.get("penalty_threshold", 2.0)   # below this → penalise
    lbd_penalty_pts  = lbd_cfg.get("penalty_points",    5.0)    # max points to deduct
    lbd_cold_thr     = lbd_cfg.get("cold_threshold",    0.5)    # below → severe penalty

    if lbd > 0:   # feature is computable (budget + duration both known)
        if lbd < lbd_cold_thr:
            # Very low value (₹1K budget / 6-month project) → severe deduction
            penalty = lbd_penalty_pts
            bd.rationale.append(
                f"[LOG-BUDGET/DUR={lbd:.2f}] Extremely low budget-to-duration ratio — "
                f"major credibility concern. Deducting {penalty:.1f} pts from structured score."
            )
        elif lbd < lbd_penalty_thr:
            # Moderately low ratio → proportional deduction
            ratio   = 1.0 - (lbd / lbd_penalty_thr)   # 0 at threshold, 1 at zero
            penalty = round(ratio * lbd_penalty_pts, 2)
            bd.rationale.append(
                f"[LOG-BUDGET/DUR={lbd:.2f}] Below-average budget-to-duration ratio. "
                f"Deducting {penalty:.1f} pts from structured score."
            )
        else:
            penalty = 0.0
            bd.rationale.append(
                f"[LOG-BUDGET/DUR={lbd:.2f}] Budget-to-duration ratio looks healthy. No penalty."
            )
        struct_total = max(0.0, round(struct_total - penalty, 2))
        bd.structured_total = struct_total

    # Step 3: semantic scoring
    semantic_scorer(lead, bd, enc, vecs, cfg)

    # Step 3.5: budget sanity gate — detects prank / contradictory leads
    budget_sanity_check(lead, bd, gate, cfg)

    # Step 3.75: absolute COLD overrides — hard-kill for NI & micro-budget pranks
    #   If triggered, bd.final_score=0, bd.tier="Cold" are already set — return early.
    hard_killed = absolute_cold_overrides(lead, bd, gate, vecs, cfg)
    if hard_killed:
        return LeadResult(lead=lead, breakdown=bd)

    # Step 4: combine and classify (respect hard cap from sanity check)
    raw_score      = struct_total + bd.semantic_total
    bd.final_score = round(min(raw_score, bd.budget_sanity_cap, 100.0), 1)
    bd.tier        = assign_tier(bd.final_score, cfg)

    # Rationale: gate explanation (after prank alert so it appears below it)
    insert_at = 1 if bd.budget_sanity_triggered else 0
    if gate < cfg.get("intent_gate", {}).get("cold_threshold", 0.35):
        bd.rationale.insert(insert_at,
            f"[GATE=COLD] Intent gate={gate:.2f} — structured scores dampened by {bd.dampen_factor:.2f}x"
        )
    elif gate >= cfg.get("intent_gate", {}).get("hot_threshold", 0.60):
        bd.rationale.insert(insert_at,
            f"[GATE=HOT] Intent gate={gate:.2f} — full weights applied"
        )
    else:
        bd.rationale.insert(insert_at,
            f"[GATE=NEUTRAL] Intent gate={gate:.2f} — normal weights applied"
        )

    return LeadResult(lead=lead, breakdown=bd)



# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round((value / max_val) * width)) if max_val else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_report(result: LeadResult) -> None:
    """Print a formatted ASCII classification report."""
    lead = result.lead
    bd   = result.breakdown
    W    = 64

    def rule(c="-"): print(c * W)
    def ln(t=""):
        try:
            print(str(t))
        except UnicodeEncodeError:
            safe = str(t).replace("₹", "Rs.").replace("—", "-")
            print(safe.encode("ascii", errors="replace").decode("ascii"))

    rule("=")
    ln("  LEAD CLASSIFICATION REPORT")
    rule("=")
    ln(f"  Lead ID    : {lead.get('lead_id', 'N/A')}")
    ln(f"  Name       : {lead.get('name', 'N/A')}")
    ln(f"  Company    : {lead.get('company_name', 'N/A')}")
    ln(f"  Industry   : {lead.get('industry', 'N/A')}")
    ln(f"  Budget     : {lead.get('estimated_budget', 'N/A')}")
    ln(f"  Duration   : {lead.get('project_duration', 'N/A')}")
    ln(f"  Urgency    : {lead.get('urgency', 'N/A')}")
    ln(f"  Source     : {lead.get('lead_source', 'N/A')}")
    ln(f"  Intent Gate: {bd.intent_gate:.2f}  {'HOT' if bd.intent_gate>=0.60 else 'COLD' if bd.intent_gate<0.35 else 'NEUTRAL'}")
    if bd.ni_override_triggered:
        ln("  *** NOT-INTERESTED HARD KILL FIRED -- score zeroed ***")
    if bd.prank_budget_triggered:
        ln("  *** PRANK BUDGET HARD KILL FIRED -- score zeroed ***")
    rule()
    ln(f"  FINAL SCORE : {bd.final_score:5.1f} / 100  {_bar(bd.final_score, 100, 32)}")
    ln(f"  TIER        : [{bd.tier.upper()}]")
    rule()
    ln("  STRUCTURED SCORE   (max ~50 after weights)")
    rule(".")
    ln(f"  {'Budget':<22} {bd.budget_score:5.1f}  {_bar(bd.budget_score, 15*1.2)}")
    ln(f"  {'Urgency':<22} {bd.urgency_score:5.1f}  {_bar(bd.urgency_score, 12)}")
    ln(f"  {'Decision Maker':<22} {bd.decision_maker_score:5.1f}  {_bar(bd.decision_maker_score, 12)}")
    ln(f"  {'Company Size':<22} {bd.company_size_score:5.1f}  {_bar(bd.company_size_score, 8)}")
    ln(f"  {'Lead Source':<22} {bd.lead_source_score:5.1f}  {_bar(bd.lead_source_score, 5)}")
    ln(f"  {'Gate Dampen':<22} {bd.dampen_factor:.2f}x")
    lbd_label = f"{bd.log_budget_per_duration:.2f}" if bd.log_budget_per_duration else "N/A"
    ln(f"  {'Log(Bdg)/Duration':<22} {lbd_label}")
    rule(".")
    ln(f"  {'Structured Total':<22} {bd.structured_total:5.1f}")
    rule()
    ln("  SEMANTIC SCORE     (max ~50 after weights)")
    rule(".")
    ln(f"  {'Intent (gate sims)':<22} {bd.intent_score:5.1f}  {_bar(bd.intent_score, 26)}")
    ln(f"  {'Clarity (desc)':<22} {bd.clarity_score:5.1f}  {_bar(bd.clarity_score, 20)}")
    ln(f"  {'Service Scope':<22} {bd.service_scope_score:5.1f}  {_bar(bd.service_scope_score, 10)}")
    rule(".")
    ln(f"  {'Semantic Total':<22} {bd.semantic_total:5.1f}")
    rule()
    ln("  AI ANALYSIS")
    rule(".")
    for r in bd.rationale:
        ln(f"  {r}")
    rule()
    ln("  RECOMMENDED ACTION")
    rule(".")
    if bd.tier == "Hot":
        ln("  [HOT]  Call within 24 hours. Assign a senior sales rep.")
        ln("         Prepare portfolio, cost estimate & timeline.")
        ln("         Schedule a discovery call / demo ASAP.")
    elif bd.tier == "Warm":
        ln("  [WARM] Follow up within 2-3 business days.")
        ln("         Send relevant case studies and capability deck.")
        ln("         Gauge exact timeline and budget on next contact.")
    else:
        ln("  [COLD] Low priority. Add to long-term nurture sequence.")
        ln("         Send automated monthly newsletter.")
        ln("         Re-evaluate if lead re-engages with specific intent.")
    rule("=")
    ln()
