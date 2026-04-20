"""
Deal Trajectory Engine — Phase 3: Stall Predictor
==================================================
Loads active pipeline deals and scores each one against the stall signature
library derived in Phase 2. Produces a triage-ready output with:

  - A match score (0–1) against each of the 4 stall signatures
  - A trajectory tag: healthy / watch / at-risk / critical
  - Signal-level evidence: exactly which thresholds were breached and by how much
  - Contradiction check: does any rep note contradict the behavioral pattern?
  - Urgency weighting: same signal in week 8 vs week 2 means very different things
  - An intervention card pulled from the Phase 2 library

Outputs
-------
  scored_pipeline.csv          — full scored dataset, one row per deal
  flagged_deals.csv            — at-risk + critical deals only, sorted by priority score
  intervention_queue.json      — prioritized action list, one card per flagged deal
  predictor_summary.json       — aggregate stats for dashboard/reporting
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
PIPELINE_PATH   = Path("/home/claude/active_pipeline.csv")
SIGNATURE_PATH  = Path("/home/claude/outputs/stall_signature_library.json")
INTERV_PATH     = Path("/home/claude/outputs/intervention_library.json")
OUTPUT_DIR      = Path("/home/claude/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Trajectory thresholds ────────────────────────────────────────────────────
# These govern how a deal's composite stall score maps to a label.
# Tuned so "critical" is a small, actionable list — not half the pipeline.
THRESHOLDS = {
    "critical":  0.80,   # ≥ 80% of defining signals breached + at least 2 signals hit
    "at_risk":   0.55,   # ≥ 55% + at least 2 signals hit
    "watch":     0.30,   # ≥ 30%
    "healthy":   0.0,    # below watch
}
MIN_SIGNALS_FOR_FLAG = 2   # must hit at least this many signals to be at-risk or critical
# Note: Exec Vacuum requires 2 binary signals — both must be hit for score = 1.0
# Stage gating prevents early-stage deals from triggering stall types prematurely

# Urgency multipliers — a stall in the back half of a deal's expected cycle
# is worth escalating faster than the same signal in week 2
URGENCY_MULTIPLIER = {
    "early":   0.8,   # < 33% through expected cycle
    "mid":     1.0,   # 33–66%
    "late":    1.3,   # > 66%
    "overdue": 1.5,   # past expected close
}

# Segment-based expected cycle lengths (days) — used for urgency weighting
EXPECTED_CYCLE = {
    "SMB":         45,
    "Mid-Market":  90,
    "Enterprise":  180,
    "Unknown":     90,
}

# Stated-reason → implied stall (mirrors Phase 2 contradiction map)
REASON_TO_STALL = {
    "Champion Left":                "Champion Collapse",
    "Chose Competitor":             "Competitor Displacement",
    "No Decision / Stalled":        "Ghost Stall",
    "Price / Budget":               "Competitor Displacement",
    "Wrong Timing":                 "Ghost Stall",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Drop duplicates
    df = df[~df["deal_id"].astype(str).str.endswith("-DUP", na=False)].copy()

    # Coerce booleans
    bool_cols = [
        "executive_sponsor_engaged", "pricing_objection_raised",
        "technical_validation_complete", "security_review_complete",
        "legal_review_started", "proposal_sent", "product_trial_active",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes")
                else (False if str(x).strip().lower() in ("false", "0", "no") else np.nan)
            ).fillna(False)

    # Coerce numerics
    numeric_cols = [
        "email_response_latency_delta_days", "email_response_rate",
        "days_since_last_rep_touch", "days_since_last_activity",
        "champion_engagement_score_delta", "champion_engagement_score",
        "meeting_count_last_30d", "unique_stakeholders_at_midpoint",
        "unique_stakeholders_total", "multithread_attempt_count",
        "days_since_multithread_touch", "economic_buyer_meetings",
        "competitive_mentions_count", "pricing_objection_count",
        "stage_regression_count", "discount_pct", "days_in_current_stage",
        "stage_change_count", "days_to_proposal", "days_to_first_exec_engagement",
        "acv_usd", "net_acv_usd", "intent_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[load] {len(df)} active pipeline deals loaded")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORE A SINGLE DEAL AGAINST A SINGLE SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

def score_deal_against_signature(row: dict, stall_type: str, sig_meta: dict) -> dict:
    """
    Returns a score dict:
      score         — 0 to 1 fraction of defining signals breached
      signals_hit   — list of breached signals with magnitude
      signals_clean — list of signals that looked healthy
      signals_missing — list of signals we couldn't evaluate (null)
      weighted_score — score × urgency multiplier
    """
    defining = sig_meta.get("defining_signals", {})
    if not defining:
        return {"score": 0.0, "weighted_score": 0.0,
                "signals_hit": [], "signals_clean": [], "signals_missing": []}

    signals_hit     = []
    signals_clean   = []
    signals_missing = []

    for signal, smeta in defining.items():
        raw = row.get(signal)

        # Handle missing
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            signals_missing.append(signal)
            continue

        try:
            val = float(raw)
        except (TypeError, ValueError):
            signals_missing.append(signal)
            continue

        threshold = smeta["threshold"]
        direction = smeta["direction"]
        healthy_med = smeta.get("healthy_median", threshold)
        stall_med   = smeta.get("stall_median", threshold)

        # How far past the threshold is this value? (0 = at threshold, 1 = at stall median)
        breach_range = abs(stall_med - threshold) if abs(stall_med - threshold) > 0 else 1.0

        if direction == "above" and val > threshold:
            magnitude = min(1.0, (val - threshold) / breach_range)
            signals_hit.append({
                "signal":    signal,
                "value":     round(val, 2),
                "threshold": threshold,
                "magnitude": round(magnitude, 3),
                "label":     _signal_label(signal, val, threshold, direction, stall_type),
            })
        elif direction == "below" and val < threshold:
            magnitude = min(1.0, (threshold - val) / breach_range)
            signals_hit.append({
                "signal":    signal,
                "value":     round(val, 2),
                "threshold": threshold,
                "magnitude": round(magnitude, 3),
                "label":     _signal_label(signal, val, threshold, direction, stall_type),
            })
        else:
            signals_clean.append(signal)

    total_evaluated = len(signals_hit) + len(signals_clean)
    score = len(signals_hit) / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "score":           round(score, 3),
        "signals_hit":     signals_hit,
        "signals_clean":   signals_clean,
        "signals_missing": signals_missing,
    }


def _signal_label(signal: str, val: float, threshold: float, direction: str, stall_type: str) -> str:
    """Human-readable one-liner for a breached signal."""
    templates = {
        "email_response_latency_delta_days":
            f"Email response time has slowed by {val:.1f} days since deal start (threshold: {threshold:.1f}d)",
        "champion_engagement_score_delta":
            f"Champion engagement dropped {abs(val):.0f} pts (threshold: {threshold:.0f})",
        "champion_engagement_score":
            f"Champion engagement at {val:.0f}/100 — below healthy baseline of {threshold:.0f}",
        "email_response_rate":
            f"Only {val*100:.0f}% of emails getting replies (threshold: {threshold*100:.0f}%)",
        "days_since_last_rep_touch":
            f"Last rep outreach was {val:.0f} days ago (threshold: {threshold:.0f}d)",
        "days_in_current_stage":
            f"Deal has been stuck in current stage for {val:.0f} days (threshold: {threshold:.0f}d)",
        "meeting_count_last_30d":
            f"Only {val:.0f} meetings in the last 30 days (threshold: {threshold:.0f})",
        "unique_stakeholders_at_midpoint":
            f"Only {val:.0f} stakeholder(s) engaged at midpoint (threshold: {threshold:.0f})",
        "days_since_multithread_touch":
            f"Last multi-thread touch was {val:.0f} days ago (threshold: {threshold:.0f}d)",
        "competitive_mentions_count":
            f"{val:.0f} competitor mentions logged (threshold: {threshold:.0f})",
        "discount_pct":
            f"Discount at {val*100:.1f}% — above threshold of {threshold*100:.1f}%",
        "stage_regression_count":
            f"Deal has regressed {val:.0f} stage(s) (threshold: {threshold:.0f})",
        "economic_buyer_meetings":
            f"Only {val:.0f} economic buyer meeting(s) (threshold: {threshold:.0f})",
        "executive_sponsor_engaged":
            f"No executive sponsor engaged (required by this stage for recovery likelihood)",
        "pricing_objection_count":
            f"{val:.0f} pricing objections raised (threshold: {threshold:.0f})",
    }
    return templates.get(signal, f"{signal} = {val:.2f} (threshold: {threshold:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. URGENCY WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_urgency(row: dict) -> tuple[float, str]:
    """
    Returns (multiplier, urgency_label) based on how far into the expected
    sales cycle the deal is. A deal in week 8 of an expected 10-week cycle
    warrants faster action than the same signal in week 2.
    """
    segment     = str(row.get("segment", "Unknown"))
    days_in_stage = row.get("days_in_current_stage")
    stage_changes = row.get("stage_change_count", 0) or 0
    expected     = EXPECTED_CYCLE.get(segment, 90)

    # Estimate total days elapsed: approximate from stage changes + time in current stage
    # Each stage change averages ~(expected / 5) days, plus current stage time
    avg_per_stage = expected / 5
    try:
        days_elapsed = float(stage_changes) * avg_per_stage + (float(days_in_stage) if days_in_stage else 0)
    except (TypeError, ValueError):
        days_elapsed = 0

    pct_through = days_elapsed / expected if expected > 0 else 0

    if pct_through > 1.0:
        return URGENCY_MULTIPLIER["overdue"], "overdue"
    elif pct_through > 0.66:
        return URGENCY_MULTIPLIER["late"], "late"
    elif pct_through > 0.33:
        return URGENCY_MULTIPLIER["mid"], "mid"
    else:
        return URGENCY_MULTIPLIER["early"], "early"


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTRADICTION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_note_contradiction(row: dict, dominant_stall: str) -> dict | None:
    """
    If rep notes or forecast category suggest a different root cause than the
    behavioral dominant stall, surface the contradiction.
    """
    rep_notes       = str(row.get("rep_notes", "") or "").lower()
    forecast_cat    = str(row.get("forecast_category", "") or "").lower()
    stated_reason   = str(row.get("stated_loss_reason", "") or "").lower()

    # Keyword signals in notes that imply a different stall type
    note_signals = {
        "Champion Collapse": ["left the company", "champion left", "new lead", "lost access",
                               "moved teams", "different team", "lost champion"],
        "Competitor Displacement": ["snowflake", "databricks", "fivetran", "dbt", "competitor",
                                     "going with", "chose another", "selected another"],
        "Ghost Stall":  ["not responding", "gone dark", "no reply", "unreachable",
                          "silent", "no response", "haven't heard"],
        "Exec Vacuum":  ["no budget authority", "can't approve", "above their level",
                          "waiting on finance", "no exec", "needs approval"],
    }

    implied_by_notes = []
    for stall_type, keywords in note_signals.items():
        if any(kw in rep_notes for kw in keywords):
            implied_by_notes.append(stall_type)

    if not implied_by_notes:
        return None
    if dominant_stall in implied_by_notes:
        return None  # Notes confirm the behavioral finding — no contradiction

    # Notes imply something different from what the signals show
    return {
        "behavioral_signal":  dominant_stall,
        "notes_imply":        implied_by_notes[0],
        "flag":               (
            f"Rep notes suggest '{implied_by_notes[0]}' but behavioral signals "
            f"point to '{dominant_stall}'. Investigate before intervening."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD PRIORITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_priority_score(row: dict, weighted_score: float, acv: float) -> float:
    """
    Composite priority score used to sort the intervention queue.
    Combines stall severity, urgency, and deal value so the highest-priority
    actions bubble to the top.

    Priority = weighted_stall_score × log(acv_factor) × urgency_multiplier
    """
    acv_factor = max(1.0, float(acv or 1)) / 10000   # normalize ~$10k baseline
    acv_weight  = min(3.0, np.log1p(acv_factor))       # log-compress, cap at 3x
    return round(weighted_score * acv_weight, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SCORE ALL DEALS
# ─────────────────────────────────────────────────────────────────────────────

def score_pipeline(df: pd.DataFrame, signature_library: dict,
                   intervention_library: dict) -> tuple[pd.DataFrame, list]:
    """
    Score every deal in the pipeline. Returns:
      scored_df      — original df with score columns appended
      intervention_queue — list of action cards for flagged deals
    """
    stall_types = list(signature_library.keys())
    rows_out    = []
    intervention_queue = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        urgency_mult, urgency_label = compute_urgency(row_dict)

        # Score against every stall type
        current_stage = str(row_dict.get("current_stage", "") or "").lower()
        stall_scores = {}
        stall_details = {}

        # Stage gates: some stall types aren't meaningful at early stages
        # Exec Vacuum requires the deal to be past Discovery before it fires
        # Champion Collapse requires at least Technical Validation stage
        stage_gates = {
            "Exec Vacuum":              ["technical validation", "proposal sent", "negotiation"],
            "Champion Collapse":        ["discovery", "technical validation", "proposal sent", "negotiation"],
            "Competitor Displacement":  ["discovery", "technical validation", "proposal sent", "negotiation"],
            "Ghost Stall":              ["discovery", "technical validation", "proposal sent", "negotiation"],
        }

        for stall_type in stall_types:
            required_stages = stage_gates.get(stall_type, [])
            # If stage gate exists and current stage is not in it, suppress score
            if required_stages and not any(s in current_stage for s in required_stages):
                stall_scores[stall_type]  = 0.0
                stall_details[stall_type] = {
                    "score": 0.0, "weighted_score": 0.0,
                    "signals_hit": [], "signals_clean": [], "signals_missing": [],
                    "suppressed": f"Stage gate: '{current_stage}' is too early for {stall_type}"
                }
                continue

            sig_meta = signature_library[stall_type]
            result   = score_deal_against_signature(row_dict, stall_type, sig_meta)
            weighted = round(min(1.0, result["score"] * urgency_mult), 3)
            stall_scores[stall_type]  = result["score"]
            stall_details[stall_type] = {**result, "weighted_score": weighted}

        # Dominant stall = highest weighted score
        dominant_stall   = max(stall_scores, key=stall_scores.get)
        dominant_score   = stall_scores[dominant_stall]
        dominant_weighted = stall_details[dominant_stall]["weighted_score"]

        # Trajectory tag
        if dominant_weighted >= THRESHOLDS["critical"]:
            trajectory = "critical"
        elif dominant_weighted >= THRESHOLDS["at_risk"]:
            trajectory = "at-risk"
        elif dominant_weighted >= THRESHOLDS["watch"]:
            trajectory = "watch"
        else:
            trajectory = "healthy"

        # Contradiction check
        contradiction = check_note_contradiction(row_dict, dominant_stall)

        # Priority score (used for sorting intervention queue)
        priority = compute_priority_score(
            row_dict, dominant_weighted, row_dict.get("acv_usd", 0)
        )

        # Pull intervention card from library
        interv_data  = intervention_library.get(dominant_stall, {})
        interventions = interv_data.get("interventions", [])
        top_interv    = interventions[0] if interventions else None

        # Build output row
        scored_row = {
            **row_dict,
            "trajectory":              trajectory,
            "dominant_stall_type":     dominant_stall,
            "dominant_stall_score":    dominant_score,
            "dominant_weighted_score": dominant_weighted,
            "urgency_stage":           urgency_label,
            "urgency_multiplier":      urgency_mult,
            "priority_score":          priority,
            "signals_hit_count":       len(stall_details[dominant_stall]["signals_hit"]),
            "signals_hit_labels":      " | ".join(
                s["label"] for s in stall_details[dominant_stall]["signals_hit"]
            ),
            "signals_missing_count":   len(stall_details[dominant_stall]["signals_missing"]),
            "note_contradiction":      contradiction["flag"] if contradiction else None,
            # Individual stall scores for the full picture
            "score_champion_collapse":       round(stall_scores.get("Champion Collapse", 0), 3),
            "score_competitor_displacement": round(stall_scores.get("Competitor Displacement", 0), 3),
            "score_ghost_stall":             round(stall_scores.get("Ghost Stall", 0), 3),
            "score_exec_vacuum":             round(stall_scores.get("Exec Vacuum", 0), 3),
        }
        rows_out.append(scored_row)

        # Build intervention queue entry for flagged deals
        if trajectory in ("at-risk", "critical"):
            queue_entry = _build_queue_entry(
                row_dict, trajectory, dominant_stall, dominant_score,
                dominant_weighted, urgency_label, priority,
                stall_details[dominant_stall],
                signature_library[dominant_stall],
                top_interv, contradiction, interv_data
            )
            intervention_queue.append(queue_entry)

    scored_df = pd.DataFrame(rows_out)
    intervention_queue.sort(key=lambda x: x["priority_score"], reverse=True)

    return scored_df, intervention_queue


def _build_queue_entry(row, trajectory, dominant_stall, score, weighted_score,
                        urgency, priority, detail, sig_meta,
                        top_interv, contradiction, interv_data) -> dict:
    """Build a full intervention card for one deal."""
    signals_hit = detail.get("signals_hit", [])

    # Find comparable won deals count from intervention library
    recovery_rate  = interv_data.get("recovery_rate", 0)
    recovered_count = interv_data.get("recovered_deal_count", 0)

    # Window statement: how many signals = how urgent
    n_hit    = len(signals_hit)
    n_total  = n_hit + len(detail.get("signals_clean", []))
    coverage = f"{n_hit}/{n_total} defining signals breached"

    return {
        "deal_id":              row.get("deal_id"),
        "company_name":         row.get("company_name"),
        "rep_name":             row.get("rep_name"),
        "segment":              row.get("segment"),
        "acv_usd":              row.get("acv_usd"),
        "current_stage":        row.get("current_stage"),
        "trajectory":           trajectory,
        "dominant_stall_type":  dominant_stall,
        "stall_score":          score,
        "weighted_score":       weighted_score,
        "urgency_stage":        urgency,
        "priority_score":       priority,
        "signal_coverage":      coverage,
        "stall_description":    sig_meta.get("description", ""),
        "evidence": [s["label"] for s in signals_hit],
        "historical_context": (
            f"In {recovered_count} comparable deals that showed this pattern, "
            f"{recovery_rate*100:.0f}% were recovered. "
            f"The intervention window is narrow — deals at this signal level "
            f"that were not acted on within ~7 days had significantly lower recovery rates."
            if recovery_rate > 0 else
            "Limited recovery history for this pattern — treat as high-risk."
        ),
        "top_intervention":     top_interv,
        "note_contradiction":   contradiction,
        "generated_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(scored_df: pd.DataFrame, intervention_queue: list) -> dict:
    """Aggregate stats for reporting and dashboard consumption."""
    total   = len(scored_df)
    traj_counts = scored_df["trajectory"].value_counts().to_dict()

    # ACV at risk
    flagged = scored_df[scored_df["trajectory"].isin(["at-risk", "critical"])]
    acv_at_risk = float(flagged["acv_usd"].dropna().sum())
    acv_total   = float(scored_df["acv_usd"].dropna().sum())

    # Stall type breakdown
    stall_breakdown = (
        scored_df[scored_df["trajectory"].isin(["at-risk", "critical"])]
        ["dominant_stall_type"].value_counts().to_dict()
    )

    # Segment breakdown
    seg_breakdown = (
        scored_df[scored_df["trajectory"].isin(["at-risk", "critical"])]
        .groupby("segment")["trajectory"].value_counts()
        .to_dict()
    )

    # Rep-level: which reps have the most flagged deals
    rep_flags = (
        scored_df[scored_df["trajectory"].isin(["at-risk", "critical"])]
        .groupby("rep_name").size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    # Contradiction count
    contradiction_count = scored_df["note_contradiction"].notna().sum()

    # Top 5 priority deals
    top5 = [
        {
            "deal_id":      q["deal_id"],
            "company":      q["company_name"],
            "acv":          q["acv_usd"],
            "stall":        q["dominant_stall_type"],
            "trajectory":   q["trajectory"],
            "priority":     q["priority_score"],
            "top_action":   q["top_intervention"]["action"] if q["top_intervention"] else "See intervention library",
        }
        for q in intervention_queue[:5]
    ]

    return {
        "run_timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_pipeline_deals":   total,
        "trajectory_distribution": traj_counts,
        "flagged_deal_count":     len(flagged),
        "flagged_pct":            round(len(flagged) / total, 3) if total > 0 else 0,
        "total_acv_scored":       round(acv_total, 0),
        "acv_at_risk":            round(acv_at_risk, 0),
        "acv_at_risk_pct":        round(acv_at_risk / acv_total, 3) if acv_total > 0 else 0,
        "stall_type_distribution": stall_breakdown,
        "segment_distribution":   {str(k): int(v) for k, v in seg_breakdown.items()},
        "reps_with_most_flags":   rep_flags,
        "contradiction_flags":    int(contradiction_count),
        "top_5_priority_deals":   top5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_predictor():
    print("\n" + "="*60)
    print("DEAL TRAJECTORY ENGINE — PHASE 3: STALL PREDICTOR")
    print("="*60 + "\n")

    # Load inputs
    print("[1] Loading pipeline and Phase 2 outputs...")
    df = load_pipeline(PIPELINE_PATH)

    with open(SIGNATURE_PATH) as f:
        signature_library = json.load(f)
    print(f"  → Signature library: {len(signature_library)} stall types")

    with open(INTERV_PATH) as f:
        intervention_library = json.load(f)
    print(f"  → Intervention library: {len(intervention_library)} stall types")

    # Score pipeline
    print("\n[2] Scoring all deals...")
    scored_df, intervention_queue = score_pipeline(df, signature_library, intervention_library)

    # Summary
    print("\n[3] Building summary...")
    summary = build_summary(scored_df, intervention_queue)

    # Print results
    print(f"\n{'─'*50}")
    print(f"  PIPELINE TRIAGE RESULTS")
    print(f"{'─'*50}")
    print(f"  Total deals scored:   {summary['total_pipeline_deals']}")
    print(f"  Trajectory breakdown:")
    for label, count in summary["trajectory_distribution"].items():
        bar = "█" * count
        print(f"    {label:<12} {count:>3}  {bar}")
    print(f"\n  ACV at risk:  ${summary['acv_at_risk']:>12,.0f}  "
          f"({summary['acv_at_risk_pct']*100:.1f}% of pipeline)")
    print(f"  Flagged deals: {summary['flagged_deal_count']} "
          f"({summary['flagged_pct']*100:.1f}% of pipeline)")
    print(f"\n  Dominant stall types in flagged deals:")
    for st, count in summary["stall_type_distribution"].items():
        print(f"    {st:<30} {count}")
    print(f"\n  Contradiction flags (notes ≠ signals): {summary['contradiction_flags']}")

    print(f"\n  TOP 5 PRIORITY DEALS:")
    print(f"  {'Deal ID':<12} {'Company':<30} {'ACV':>10}  {'Stall Type':<25} {'Action'}")
    print(f"  {'─'*115}")
    for deal in summary["top_5_priority_deals"]:
        company  = str(deal["company"] or "")[:28]
        action   = str(deal["top_action"] or "")[:50]
        acv      = f"${deal['acv']:>9,.0f}" if deal["acv"] else "  Unknown  "
        print(f"  {str(deal['deal_id']):<12} {company:<30} {acv}  {deal['stall']:<25} {action}")

    # Write outputs
    print(f"\n[4] Writing output files...")

    scored_df.to_csv(OUTPUT_DIR / "scored_pipeline.csv", index=False)
    print(f"  → scored_pipeline.csv ({len(scored_df)} rows)")

    flagged = scored_df[scored_df["trajectory"].isin(["at-risk", "critical"])].sort_values(
        "priority_score", ascending=False
    )
    flagged.to_csv(OUTPUT_DIR / "flagged_deals.csv", index=False)
    print(f"  → flagged_deals.csv ({len(flagged)} rows)")

    with open(OUTPUT_DIR / "intervention_queue.json", "w") as f:
        json.dump(intervention_queue, f, indent=2, default=str)
    print(f"  → intervention_queue.json ({len(intervention_queue)} cards)")

    with open(OUTPUT_DIR / "predictor_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → predictor_summary.json")

    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)

    return scored_df, intervention_queue, summary


if __name__ == "__main__":
    scored_df, queue, summary = run_predictor()
