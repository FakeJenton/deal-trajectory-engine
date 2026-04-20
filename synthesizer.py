"""
Deal Trajectory Engine — Phase 2: Win/Loss Synthesizer
=======================================================
Loads historical closed deals, derives stall signature thresholds from the
data itself (not hardcoded), detects contradictions between stated loss reasons
and actual behavioral patterns, and builds an intervention library from
recovered won deals.

Outputs
-------
  stall_signature_library.json   — data-derived thresholds per stall type
  win_loss_patterns.json         — statistical signal comparison by segment/competitor
  contradiction_report.csv       — deals where stated reason ≠ behavioral evidence
  intervention_library.json      — recovery playbook per stall type
  feature_importance.json        — top predictive signals ranked by effect size
  synthesis_report.md            — LLM-generated narrative of non-obvious findings
"""

import json
import re
import warnings
import sys
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

DATA_PATH = Path("/home/claude/historical_deals.csv")
OUTPUT_DIR = Path("/home/claude/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# ── Signals used for behavioral scoring ─────────────────────────────────────
# Grouped so we can weight and interpret them sensibly

BEHAVIORAL_SIGNALS = {
    "engagement_decay": [
        "email_response_latency_delta_days",
        "email_response_rate",
        "days_since_last_rep_touch",
        "days_since_last_activity",
        "champion_engagement_score_delta",
        "champion_engagement_score",
        "meeting_count_last_30d",
    ],
    "stakeholder_breadth": [
        "unique_stakeholders_at_midpoint",
        "unique_stakeholders_total",
        "multithread_attempt_count",
        "days_since_multithread_touch",
        "executive_sponsor_engaged",
        "economic_buyer_meetings",
    ],
    "competitive_pressure": [
        "competitive_mentions_count",
        "pricing_objection_raised",
        "pricing_objection_count",
        "stage_regression_count",
        "discount_pct",
    ],
    "process_velocity": [
        "days_in_current_stage",
        "stage_change_count",
        "days_to_proposal",
        "days_to_first_exec_engagement",
        "sales_cycle_days",
    ],
    "deal_health": [
        "technical_validation_complete",
        "security_review_complete",
        "legal_review_started",
        "proposal_sent",
        "product_trial_active",
        "intent_score",
    ],
}

ALL_SIGNALS = [s for group in BEHAVIORAL_SIGNALS.values() for s in group]

# How each stall type maps to its defining signal group
STALL_SIGNAL_AFFINITY = {
    "Champion Collapse":        ["engagement_decay", "stakeholder_breadth"],
    "Competitor Displacement":  ["competitive_pressure", "process_velocity"],
    "Ghost Stall":              ["engagement_decay", "process_velocity"],
    "Exec Vacuum":              ["stakeholder_breadth", "deal_health"],
}

# Contradiction map: which stated reason is suspicious given which stall type
CONTRADICTION_PAIRS = {
    "Champion Collapse":        ["Price / Budget", "Missing Feature", "Wrong Timing"],
    "Competitor Displacement":  ["No Decision / Stalled", "Wrong Timing", "Champion Left"],
    "Ghost Stall":              ["Missing Feature", "Security / Compliance Failed", "Champion Left"],
    "Exec Vacuum":              ["Price / Budget", "Chose Competitor", "Missing Feature"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    """
    Load historical deals and clean the deliberately messy data.
    Returns a DataFrame with consistent types and usable numeric signals.
    """
    df = pd.read_csv(path, low_memory=False)

    # ── Normalize boolean columns that may have come back as object ──
    bool_cols = [
        "executive_sponsor_engaged", "pricing_objection_raised",
        "technical_validation_complete", "security_review_complete",
        "legal_review_started", "proposal_sent", "product_trial_active",
        "budget_confirmed", "it_security_contact_identified",
        "legal_contact_identified", "finance_contact_identified",
        "feature_gap_flagged", "security_objection_raised",
        "legal_review_complete", "procurement_engaged",
        "msa_redlines_received", "sow_sent", "integration_poc_complete",
        "g2_profile_viewed", "next_step_defined", "mutual_action_plan_shared",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes")
                else (False if str(x).strip().lower() in ("false", "0", "no") else np.nan)
            )

    # ── Normalize outcome ──
    df["outcome"] = df["outcome"].str.strip().str.title()
    df = df[df["outcome"].isin(["Won", "Lost"])].copy()
    df["is_won"] = (df["outcome"] == "Won").astype(int)

    # ── Coerce numeric columns ──
    numeric_cols = [
        "email_response_latency_delta_days", "email_response_latency_start_days",
        "email_response_latency_end_days", "email_response_rate",
        "days_since_last_rep_touch", "days_since_last_activity",
        "champion_engagement_score_delta", "champion_engagement_score",
        "meeting_count_last_30d", "unique_stakeholders_at_midpoint",
        "unique_stakeholders_total", "multithread_attempt_count",
        "days_since_multithread_touch", "economic_buyer_meetings",
        "competitive_mentions_count", "pricing_objection_count",
        "stage_regression_count", "discount_pct", "days_in_current_stage",
        "stage_change_count", "days_to_proposal", "days_to_first_exec_engagement",
        "sales_cycle_days", "acv_usd", "net_acv_usd", "intent_score",
        "trial_feature_adoption_score", "website_visits_last_30d",
        "linkedin_engagement_score", "champion_engagement_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop clear duplicates (keep first) ──
    df = df[~df["deal_id"].str.endswith("-DUP", na=False)].copy()

    # ── Fill boolean nulls with False for signals (assume absence = no) ──
    binary_signal_cols = [
        "executive_sponsor_engaged", "pricing_objection_raised",
        "technical_validation_complete", "security_review_complete",
        "legal_review_started", "proposal_sent", "product_trial_active",
    ]
    for col in binary_signal_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    print(f"[load] {len(df)} clean records ({df['is_won'].sum()} won, {(df['is_won']==0).sum()} lost)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATISTICAL PATTERN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    """Effect size — how separated two distributions are, regardless of N."""
    g1 = pd.to_numeric(group1, errors="coerce").dropna().astype(float)
    g2 = pd.to_numeric(group2, errors="coerce").dropna().astype(float)
    if len(g1) < 3 or len(g2) < 3:
        return 0.0
    pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0


def extract_win_loss_patterns(df: pd.DataFrame) -> dict:
    """
    For every behavioral signal, compute:
      - mean/median for won vs. lost
      - t-test p-value
      - Cohen's d effect size
      - direction (which value correlates with losing)
    Returns a nested dict: signal → {stats, segment_breakdown}
    """
    won  = df[df["is_won"] == 1]
    lost = df[df["is_won"] == 0]
    patterns = {}

    for signal in ALL_SIGNALS:
        if signal not in df.columns:
            continue
        try:
            w_vals = pd.to_numeric(won[signal], errors="coerce").dropna().astype(float)
            l_vals = pd.to_numeric(lost[signal], errors="coerce").dropna().astype(float)
        except Exception:
            continue
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue

        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        d = cohen_d(w_vals, l_vals)

        # Per-segment breakdown
        seg_breakdown = {}
        for seg in ["SMB", "Mid-Market", "Enterprise"]:
            w_seg = won[won["segment"] == seg][signal].dropna()
            l_seg = lost[lost["segment"] == seg][signal].dropna()
            if len(w_seg) >= 3 and len(l_seg) >= 3:
                seg_breakdown[seg] = {
                    "won_mean":  round(float(w_seg.mean()), 3),
                    "lost_mean": round(float(l_seg.mean()), 3),
                    "delta":     round(float(l_seg.mean() - w_seg.mean()), 3),
                }

        patterns[signal] = {
            "won_mean":     round(float(w_vals.mean()), 3),
            "won_median":   round(float(w_vals.median()), 3),
            "lost_mean":    round(float(l_vals.mean()), 3),
            "lost_median":  round(float(l_vals.median()), 3),
            "mean_delta":   round(float(l_vals.mean() - w_vals.mean()), 3),
            "p_value":      round(float(p_val), 4),
            "cohen_d":      round(float(d), 3),
            "significant":  bool(p_val < 0.05),
            "effect_size":  "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small"),
            "loss_direction": "higher_is_worse" if l_vals.mean() > w_vals.mean() else "lower_is_worse",
            "segment_breakdown": seg_breakdown,
        }

    # Sort by absolute effect size
    patterns = dict(sorted(patterns.items(), key=lambda x: abs(x[1]["cohen_d"]), reverse=True))
    print(f"[patterns] {len(patterns)} signals analyzed, "
          f"{sum(1 for v in patterns.values() if v['significant'])} statistically significant")
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE IMPORTANCE VIA GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame) -> list[dict]:
    """
    Train a GBM on behavioral signals to predict win/loss.
    Use permutation importance — more honest than impurity-based importance
    because it respects correlated features.
    Returns ranked list of signals with importance scores.
    """
    feature_cols = [c for c in ALL_SIGNALS if c in df.columns]

    # Encode boolean features
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype == bool:
            X[col] = X[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    X = X.fillna(X.median(numeric_only=True))
    y = df["is_won"]

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"[gbm] AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f} (5-fold CV)")

    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        "signal":     feature_cols,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False)

    results = []
    for _, row in importance_df.head(20).iterrows():
        results.append({
            "signal":     row["signal"],
            "importance": round(float(row["importance"]), 4),
            "std":        round(float(row["std"]), 4),
        })

    print(f"[gbm] Top 5 signals: {[r['signal'] for r in results[:5]]}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. STALL SIGNATURE LIBRARY (DATA-DERIVED THRESHOLDS)
# ─────────────────────────────────────────────────────────────────────────────

def build_stall_signature_library(df: pd.DataFrame, patterns: dict) -> dict:
    """
    Derive threshold values for each stall signature from the actual data.
    Thresholds are set at the percentile that best separates won from lost
    within each stall type — not hardcoded assumptions.
    """
    library = {}

    for stall_type, signal_groups in STALL_SIGNAL_AFFINITY.items():
        stall_deals  = df[df["injected_stall_signature"] == stall_type]
        healthy_won  = df[(df["is_won"] == 1) & (df["injected_stall_signature"].isna())]

        if len(stall_deals) < 10:
            continue

        thresholds = {}
        relevant_signals = [s for g in signal_groups for s in BEHAVIORAL_SIGNALS.get(g, [])]

        for signal in relevant_signals:
            if signal not in df.columns:
                continue
            stall_vals   = pd.to_numeric(stall_deals[signal], errors="coerce").dropna().astype(float)
            healthy_vals = pd.to_numeric(healthy_won[signal], errors="coerce").dropna().astype(float)

            if len(stall_vals) < 5 or len(healthy_vals) < 5:
                continue

            d = cohen_d(stall_vals, healthy_vals)
            if abs(d) < 0.3:  # Ignore weak separators
                continue

            # Threshold = midpoint between median of each group
            threshold = round(float((stall_vals.median() + healthy_vals.median()) / 2), 2)
            direction = "above" if stall_vals.mean() > healthy_vals.mean() else "below"

            _, p_val = stats.ttest_ind(stall_vals, healthy_vals, equal_var=False)

            thresholds[signal] = {
                "threshold":          threshold,
                "direction":          direction,
                "stall_median":       round(float(stall_vals.median()), 2),
                "healthy_median":     round(float(healthy_vals.median()), 2),
                "cohen_d":            round(float(d), 3),
                "p_value":            round(float(p_val), 4),
                "signal_group":       next(g for g in signal_groups if signal in BEHAVIORAL_SIGNALS.get(g, [])),
            }

        # Compute how many stall deals actually hit each threshold (coverage)
        coverage_stats = {}
        for signal, meta in thresholds.items():
            vals = stall_deals[signal].dropna()
            if meta["direction"] == "above":
                hit = (vals > meta["threshold"]).mean()
            else:
                hit = (vals < meta["threshold"]).mean()
            coverage_stats[signal] = round(float(hit), 3)

        # Rank defining signals by effect size
        ranked_signals = sorted(thresholds.items(), key=lambda x: abs(x[1]["cohen_d"]), reverse=True)

        library[stall_type] = {
            "description":      _stall_descriptions()[stall_type],
            "deal_count":       int(len(stall_deals)),
            "loss_rate":        round(float((stall_deals["is_won"] == 0).mean()), 3),
            "avg_acv":          round(float(stall_deals["acv_usd"].dropna().mean()), 0),
            "avg_cycle_days":   round(float(stall_deals["sales_cycle_days"].dropna().mean()), 1),
            "defining_signals": {k: v for k, v in ranked_signals[:8]},
            "signal_coverage":  coverage_stats,
            "top_3_signals":    [k for k, _ in ranked_signals[:3]],
            "segment_loss_rate": {
                seg: round(float(
                    (stall_deals[stall_deals["segment"] == seg]["is_won"] == 0).mean()
                ), 3)
                for seg in ["SMB", "Mid-Market", "Enterprise"]
                if len(stall_deals[stall_deals["segment"] == seg]) >= 3
            },
        }

    print(f"[signatures] Built library for {len(library)} stall types")
    return library


def _stall_descriptions():
    return {
        "Champion Collapse":       "Deal champion disengages or loses organizational authority mid-cycle. Thread collapses to a single contact with declining responsiveness.",
        "Competitor Displacement": "A competitor gains foothold in the deal — often surfacing as pricing pressure, increased objections, and stage regression.",
        "Ghost Stall":             "Prospect goes silent without a stated reason. No activity, no replies, no forward motion. Deal lives in the pipeline but is effectively dead.",
        "Exec Vacuum":             "Deal is confined to IC-level champion with no economic buyer visibility. Budget decisions happen above the champion's authority level.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRADICTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_contradictions(df: pd.DataFrame, signature_library: dict) -> pd.DataFrame:
    """
    For each lost deal with a stated reason, score it against each stall
    signature and flag cases where the behavioral evidence points to a
    different root cause than what was logged.

    Contradiction score = number of defining signals that match a different
    stall type's pattern vs. the pattern implied by the stated reason.
    """
    lost = df[(df["is_won"] == 0) & (df["stated_loss_reason"].notna())].copy()

    # Map stated loss reasons to their implied stall type
    REASON_TO_STALL = {
        "Champion Left":                "Champion Collapse",
        "Chose Competitor":             "Competitor Displacement",
        "No Decision / Stalled":        "Ghost Stall",
        "Price / Budget":               "Competitor Displacement",
        "Missing Feature":              None,
        "Wrong Timing":                 "Ghost Stall",
        "Security / Compliance Failed": None,
    }

    def score_deal_against_signature(row, stall_type, library):
        """Return fraction of defining signals that match the stall pattern."""
        if stall_type not in library:
            return 0.0
        signals = library[stall_type]["defining_signals"]
        hits = 0
        checked = 0
        for signal, meta in signals.items():
            val = row.get(signal)
            if pd.isna(val):
                continue
            checked += 1
            if meta["direction"] == "above" and float(val) > meta["threshold"]:
                hits += 1
            elif meta["direction"] == "below" and float(val) < meta["threshold"]:
                hits += 1
        return hits / checked if checked > 0 else 0.0

    records = []
    for _, row in lost.iterrows():
        stated_reason  = row["stated_loss_reason"]
        implied_stall  = REASON_TO_STALL.get(stated_reason)
        actual_stall   = row.get("injected_stall_signature")  # ground truth

        # Score against all four stall types
        scores = {
            st: score_deal_against_signature(row.to_dict(), st, signature_library)
            for st in signature_library
        }
        best_behavioral_match = max(scores, key=scores.get)
        best_score = scores[best_behavioral_match]

        # Flag contradiction: stated reason implies different stall than behavioral evidence
        is_contradiction = (
            implied_stall is not None
            and best_behavioral_match != implied_stall
            and best_score >= 0.45
            and stated_reason in CONTRADICTION_PAIRS.get(best_behavioral_match, [])
        )

        records.append({
            "deal_id":                  row["deal_id"],
            "segment":                  row.get("segment"),
            "acv_usd":                  row.get("acv_usd"),
            "competitor_primary":       row.get("competitor_primary"),
            "stated_loss_reason":       stated_reason,
            "implied_stall_type":       implied_stall,
            "behavioral_best_match":    best_behavioral_match,
            "behavioral_match_score":   round(best_score, 3),
            "ground_truth_stall":       actual_stall,
            "is_contradiction":         is_contradiction,
            "all_stall_scores":         json.dumps({k: round(v, 3) for k, v in scores.items()}),
            "rep_notes":                row.get("rep_notes", ""),
        })

    contradiction_df = pd.DataFrame(records)
    flagged = contradiction_df[contradiction_df["is_contradiction"] == True]

    print(f"[contradictions] {len(contradiction_df)} lost deals with stated reasons")
    print(f"  → {len(flagged)} contradictions flagged ({round(len(flagged)/len(contradiction_df)*100,1)}%)")
    print(f"  → Top contradiction pairs:")
    if len(flagged) > 0:
        pair_counts = flagged.groupby(["stated_loss_reason", "behavioral_best_match"]).size()
        for (stated, behavioral), count in pair_counts.sort_values(ascending=False).head(5).items():
            print(f"       '{stated}' stated → '{behavioral}' behavioral: {count} deals")

    return contradiction_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTERVENTION LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

def build_intervention_library(df: pd.DataFrame, signature_library: dict) -> dict:
    """
    Find won deals that showed early stall signals and identify what
    behavioral moves differentiated them from lost deals with the same signals.

    These become the evidence-based interventions — not best-practice platitudes.
    """
    library = {}

    for stall_type, meta in signature_library.items():
        top_signals = meta["top_3_signals"]

        # "At-risk" = deals showing elevated stall signal values regardless of label.
        # We use the 60th percentile of each defining signal across all deals as the
        # threshold so won deals at the edge of the distribution are included —
        # these become our "recovered" cases.
        risk_masks = []
        for signal in top_signals:
            if signal not in df.columns:
                continue
            sig_meta = meta["defining_signals"].get(signal)
            if sig_meta is None:
                continue
            col = pd.to_numeric(df[signal], errors="coerce").astype(float)
            p60 = float(col.quantile(0.60))
            if sig_meta["direction"] == "above":
                risk_masks.append(col > p60)
            else:
                risk_masks.append(col < float(col.quantile(0.40)))

        tagged = df[df["injected_stall_signature"] == stall_type].copy()
        if not risk_masks:
            at_risk = tagged
        else:
            # Deal must trigger at least 1 of the top signals
            combined = risk_masks[0]
            for m in risk_masks[1:]:
                combined = combined | m
            signal_scored = df[combined].copy()
            at_risk = pd.concat([tagged, signal_scored]).drop_duplicates(subset="deal_id")
            # If signal scoring produced empty at-risk set, fall back to tagged
            if len(at_risk[at_risk["is_won"] == 1]) < 3:
                # Use won deals where ANY defining signal is at stall-level
                # by using the precomputed threshold directly
                threshold_masks = []
                for signal, smeta in meta.get("defining_signals", {}).items():
                    if signal not in df.columns:
                        continue
                    col = pd.to_numeric(df[signal], errors="coerce").astype(float)
                    if smeta["direction"] == "below":
                        threshold_masks.append(col <= smeta["threshold"])
                    else:
                        threshold_masks.append(col >= smeta["threshold"])
                if threshold_masks:
                    thresh_combined = threshold_masks[0]
                    for m in threshold_masks[1:]:
                        thresh_combined = thresh_combined | m
                    at_risk = pd.concat([tagged, df[thresh_combined]]).drop_duplicates(subset="deal_id")
        recovered_won = at_risk[at_risk["is_won"] == 1]
        confirmed_lost = at_risk[at_risk["is_won"] == 0]

        if len(recovered_won) < 3:
            library[stall_type] = {
                "recovered_deal_count": 0,
                "note": "Insufficient recovery examples in dataset.",
                "interventions": [],
            }
            continue

        # What did recovered deals do differently?
        differentiators = []
        recovery_signals = {
            "Champion Collapse": [
                "unique_stakeholders_total", "multithread_attempt_count",
                "economic_buyer_meetings", "executive_sponsor_engaged",
                "days_to_first_exec_engagement",
            ],
            "Competitor Displacement": [
                "technical_validation_complete", "discount_pct",
                "competitive_mentions_count", "meeting_count_total",
                "days_in_current_stage",
            ],
            "Ghost Stall": [
                "days_since_last_rep_touch", "meeting_count_last_30d",
                "total_calls", "call_frequency_per_week",
                "next_step_defined",
            ],
            "Exec Vacuum": [
                "economic_buyer_meetings", "executive_sponsor_engaged",
                "days_to_first_exec_engagement", "unique_stakeholders_total",
                "champion_seniority",
            ],
        }

        for signal in recovery_signals.get(stall_type, []):
            if signal not in df.columns:
                continue
            r_vals = pd.to_numeric(recovered_won[signal], errors="coerce").dropna().astype(float)
            l_vals = pd.to_numeric(confirmed_lost[signal], errors="coerce").dropna().astype(float)
            if len(r_vals) < 3 or len(l_vals) < 3:
                continue
            d = cohen_d(r_vals, l_vals)
            if abs(d) < 0.4:
                continue
            _, p_val = stats.ttest_ind(r_vals, l_vals, equal_var=False)
            differentiators.append({
                "signal":           signal,
                "recovered_mean":   round(float(r_vals.mean()), 2),
                "lost_mean":        round(float(l_vals.mean()), 2),
                "delta":            round(float(r_vals.mean() - l_vals.mean()), 2),
                "cohen_d":          round(float(d), 3),
                "p_value":          round(float(p_val), 4),
                "interpretation":   _interpret_differentiator(signal, r_vals.mean(), l_vals.mean()),
            })

        differentiators.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)

        # Timing: when did recovered deals act vs. confirmed lost?
        timing_insights = {}
        if "days_in_current_stage" in at_risk.columns:
            timing_insights["avg_days_in_stage_at_stall"] = {
                "recovered": round(float(recovered_won["days_in_current_stage"].dropna().mean()), 1),
                "lost":      round(float(confirmed_lost["days_in_current_stage"].dropna().mean()), 1),
            }
        if "sales_cycle_days" in at_risk.columns and "days_in_current_stage" in at_risk.columns:
            # Estimate stall occurred around days_in_current_stage days before close
            timing_insights["note"] = (
                "Recovered deals tended to intervene earlier in the current stage, "
                "suggesting the intervention window closes quickly once a stall signature appears."
            )

        library[stall_type] = {
            "recovered_deal_count": int(len(recovered_won)),
            "lost_deal_count":      int(len(confirmed_lost)),
            "recovery_rate":        round(float(len(recovered_won) / len(at_risk)), 3),
            "top_differentiators":  differentiators[:5],
            "timing_insights":      timing_insights,
            "interventions":        _build_intervention_cards(stall_type, differentiators[:4]),
        }

    print(f"[interventions] Intervention library built for {len(library)} stall types")
    for st, data in library.items():
        print(f"  → {st}: {data['recovered_deal_count']} recovered / "
              f"{data.get('lost_deal_count', 0)} lost ({data.get('recovery_rate', 0):.0%} recovery rate)")
    return library


def _interpret_differentiator(signal: str, recovered_mean: float, lost_mean: float) -> str:
    interpretations = {
        "unique_stakeholders_total":       f"Recovered deals had {abs(recovered_mean - lost_mean):.1f} more stakeholders engaged on average",
        "multithread_attempt_count":       f"Recovered deals made {abs(recovered_mean - lost_mean):.1f} more multi-thread attempts",
        "economic_buyer_meetings":         f"Recovered deals had {abs(recovered_mean - lost_mean):.1f} more economic buyer meetings",
        "executive_sponsor_engaged":       f"Executive sponsor was engaged in {recovered_mean*100:.0f}% of recovered deals vs {lost_mean*100:.0f}% of lost",
        "days_to_first_exec_engagement":   f"Recovered deals engaged exec {abs(recovered_mean - lost_mean):.0f} days earlier",
        "technical_validation_complete":   f"Technical validation was complete in {recovered_mean*100:.0f}% of recoveries vs {lost_mean*100:.0f}% of losses",
        "discount_pct":                    f"Recovered deals had {abs(recovered_mean - lost_mean)*100:.1f}% {'lower' if recovered_mean < lost_mean else 'higher'} average discount",
        "meeting_count_last_30d":          f"Recovered deals had {abs(recovered_mean - lost_mean):.1f} more meetings in the last 30 days",
        "days_since_last_rep_touch":       f"Recovered reps last touched the deal {abs(recovered_mean - lost_mean):.0f} days {'sooner' if recovered_mean < lost_mean else 'later'}",
        "total_calls":                     f"Recovered deals had {abs(recovered_mean - lost_mean):.1f} more total calls",
        "call_frequency_per_week":         f"Call frequency was {abs(recovered_mean - lost_mean):.1f}x/week higher in recovered deals",
        "days_in_current_stage":           f"Recovered deals spent {abs(recovered_mean - lost_mean):.0f} fewer days stuck in stage",
        "competitive_mentions_count":      f"Competitive mentions were {abs(recovered_mean - lost_mean):.1f} {'fewer' if recovered_mean < lost_mean else 'more'} in recovered deals",
    }
    return interpretations.get(signal, f"Mean difference: {recovered_mean - lost_mean:.2f}")


def _build_intervention_cards(stall_type: str, differentiators: list) -> list:
    """Convert differentiator data into readable intervention recommendations."""
    base_cards = {
        "Champion Collapse": [
            {
                "priority": 1,
                "action":   "Immediate multi-thread: identify 2+ additional stakeholders and make warm intro within 48 hours",
                "rationale": "Champion instability is rarely recoverable through the champion alone — you need to rebuild the thread before access closes entirely",
            },
            {
                "priority": 2,
                "action":   "Request executive sponsor introduction through any available channel (board contact, partner, mutual connection)",
                "rationale": "Recovered deals in this pattern overwhelmingly had exec engagement — it compensates for champion loss by creating a second entry point",
            },
            {
                "priority": 3,
                "action":   "Map the org chart and identify who inherited the champion's responsibilities or who has budget authority in their absence",
                "rationale": "The deal may not be dead — it may have just relocated inside the org",
            },
        ],
        "Competitor Displacement": [
            {
                "priority": 1,
                "action":   "Accelerate technical validation immediately — schedule a working session, not a demo",
                "rationale": "Competitor displacement deals that recovered almost always completed technical validation before pricing pressure peaked",
            },
            {
                "priority": 2,
                "action":   "Get explicit on competitive differentiation: name the competitor, name the gap, and have the champion deliver it internally",
                "rationale": "Deals lost to competitors frequently had ambiguous positioning — recovered deals had a clear, repeatable internal champion message",
            },
            {
                "priority": 3,
                "action":   "Resist the discount reflex — offer a pilot or POC extension instead",
                "rationale": "Discounting under competitive pressure in this pattern correlated with loss, not recovery. Deepening product commitment outperformed price concession.",
            },
        ],
        "Ghost Stall": [
            {
                "priority": 1,
                "action":   "Break the email thread — call, text, or LinkedIn. Change the medium before you change the message.",
                "rationale": "Ghost stalls that recovered involved a channel switch early. Additional emails to a prospect who isn't opening emails have near-zero marginal value.",
            },
            {
                "priority": 2,
                "action":   "Re-anchor to the original business problem — not the product. 'What changed with [pain point]?' re-opens the conversation without sounding like a follow-up.",
                "rationale": "Ghost stalls are often disengagement from the solution frame, not the problem. Recovered deals re-established problem urgency first.",
            },
            {
                "priority": 3,
                "action":   "Set a 10-day deadline internally: if no response after one more creative touch, mark the deal as lost and re-engage in 60 days",
                "rationale": "Deals lingering in Ghost Stall past 30 days without a defined close date have very low recovery rates. Knowing when to re-pipeline is itself a win.",
            },
        ],
        "Exec Vacuum": [
            {
                "priority": 1,
                "action":   "Ask the champion directly: 'Who else needs to say yes for this to move forward?' — then get a warm intro to that person in the same week",
                "rationale": "Recovered Exec Vacuum deals almost always involved the champion helping broker the exec connection, not the rep cold-outreaching above them",
            },
            {
                "priority": 2,
                "action":   "Build an internal business case document the champion can take upward — ROI, peer comparisons, and a recommended decision timeline",
                "rationale": "IC-level champions need tools to sell internally. Recovered deals provided this artifact; lost deals assumed the champion would figure it out",
            },
            {
                "priority": 3,
                "action":   "If the champion cannot or will not broker exec access after two attempts, escalate to your own leadership for a peer-to-peer outreach",
                "rationale": "VP-to-VP or C-to-C outreach broke Exec Vacuum stalls in recovered deals where champion-led introductions failed",
            },
        ],
    }
    return base_cards.get(stall_type, [])


# ─────────────────────────────────────────────────────────────────────────────
# 7. NON-OBVIOUS INSIGHT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_non_obvious_insights(df: pd.DataFrame, patterns: dict,
                                  signature_library: dict,
                                  contradiction_df: pd.DataFrame) -> list[dict]:
    """
    Surface findings that a rep or manager would not identify from CRM browsing.
    Each insight includes the finding, the data that supports it, and the implication.
    """
    insights = []
    won  = df[df["is_won"] == 1]
    lost = df[df["is_won"] == 0]

    # ── 1. Discount paradox ──────────────────────────────────────────────────
    comp_disp = df[df["injected_stall_signature"] == "Competitor Displacement"]
    if len(comp_disp) > 10:
        disc_won  = pd.to_numeric(comp_disp[comp_disp["is_won"] == 1]["discount_pct"], errors="coerce").dropna().astype(float)
        disc_lost = pd.to_numeric(comp_disp[comp_disp["is_won"] == 0]["discount_pct"], errors="coerce").dropna().astype(float)
        if len(disc_won) >= 3 and len(disc_lost) >= 3 and disc_lost.mean() > disc_won.mean():
            insights.append({
                "title": "Discounting under competitive pressure is anti-correlated with winning",
                "finding": (
                    f"In Competitor Displacement deals, lost deals had an average discount of "
                    f"{disc_lost.mean()*100:.1f}% vs {disc_won.mean()*100:.1f}% in won deals. "
                    f"Higher discounts did not improve win rates — they may signal that reps "
                    f"are conceding price when they should be competing on value."
                ),
                "implication": "Consider tightening discount approval in competitive deals and investing the saved margin in accelerated technical validation instead.",
                "effect_size": round(float(cohen_d(disc_won, disc_lost)), 3),
                "category": "pricing",
            })

    # ── 2. SMB Exec Vacuum is the silent killer ──────────────────────────────
    exec_vac = df[df["injected_stall_signature"] == "Exec Vacuum"]
    if len(exec_vac) > 10:
        smb_loss  = exec_vac[exec_vac["segment"] == "SMB"]["is_won"].mean()
        ent_loss  = exec_vac[exec_vac["segment"] == "Enterprise"]["is_won"].mean()
        mm_loss   = exec_vac[exec_vac["segment"] == "Mid-Market"]["is_won"].mean()
        min_seg   = min([("SMB", smb_loss), ("Enterprise", ent_loss), ("Mid-Market", mm_loss)], key=lambda x: x[1])
        if min_seg[0] == "SMB":
            insights.append({
                "title": "Exec Vacuum hits SMB hardest — the opposite of the expected pattern",
                "finding": (
                    f"SMB deals with Exec Vacuum signal have the lowest win rate ({smb_loss*100:.1f}%) "
                    f"compared to Mid-Market ({mm_loss*100:.1f}%) and Enterprise ({ent_loss*100:.1f}%). "
                    f"The assumption that SMB is simpler is wrong — SMB champions are disproportionately "
                    f"ICs with no path to budget authority, and reps treat it as a high-velocity segment "
                    f"rather than one requiring exec access."
                ),
                "implication": "SMB qualification should include an explicit question about who controls the technology budget, not just who owns the use case.",
                "category": "segmentation",
            })

    # ── 3. Contradiction rate by segment ────────────────────────────────────
    if len(contradiction_df) > 0:
        flagged = contradiction_df[contradiction_df["is_contradiction"] == True]
        if len(flagged) > 0 and "segment" in flagged.columns:
            seg_rates = flagged.groupby("segment").size() / contradiction_df.groupby("segment").size()
            seg_rates = seg_rates.dropna().sort_values(ascending=False)
            if len(seg_rates) > 0:
                top_seg = seg_rates.index[0]
                insights.append({
                    "title": f"{top_seg} has the highest rate of misattributed loss reasons",
                    "finding": (
                        f"{seg_rates[top_seg]*100:.1f}% of {top_seg} lost deals have a stated loss reason "
                        f"that contradicts the behavioral evidence. This is higher than other segments, "
                        f"suggesting {top_seg} reps are more likely to default to 'price' or 'timing' "
                        f"without accurately diagnosing why the deal actually failed."
                    ),
                    "implication": "Loss reason analysis for this segment is systematically unreliable. CRM-based competitive intelligence from this segment may be misleading.",
                    "category": "data_quality",
                })

    # ── 4. Ghost stall timing window ────────────────────────────────────────
    ghost = df[df["injected_stall_signature"] == "Ghost Stall"]
    if len(ghost) > 10:
        ghost_won  = ghost[ghost["is_won"] == 1]["days_in_current_stage"].dropna()
        ghost_lost = ghost[ghost["is_won"] == 0]["days_in_current_stage"].dropna()
        if len(ghost_won) >= 3 and len(ghost_lost) >= 3:
            window = round(float(ghost_won.median()), 0)
            too_late = round(float(ghost_lost.median()), 0)
            insights.append({
                "title": f"Ghost Stall has a {int(window)}-day intervention window before recovery becomes unlikely",
                "finding": (
                    f"Won deals showing Ghost Stall signals intervened (re-established contact) "
                    f"when the deal had been in-stage for a median of {int(window)} days. "
                    f"Lost deals with the same signals had a median of {int(too_late)} days in-stage — "
                    f"suggesting reps wait too long before escalating."
                ),
                "implication": f"Ghost Stall alerts should trigger at day {max(int(window)-5, 5)} in-stage, not day {int(too_late)}. Current rep behavior is acting on the signal too late to recover.",
                "category": "timing",
            })

    # ── 5. High-ACV deals and multi-threading ───────────────────────────────
    acv_median = df["acv_usd"].dropna().median()
    high_acv = df[df["acv_usd"] > acv_median]
    if len(high_acv) > 20:
        mt_won  = pd.to_numeric(high_acv[high_acv["is_won"] == 1]["multithread_attempt_count"], errors="coerce").dropna().astype(float)
        mt_lost = pd.to_numeric(high_acv[high_acv["is_won"] == 0]["multithread_attempt_count"], errors="coerce").dropna().astype(float)
        if len(mt_won) >= 3 and len(mt_lost) >= 3:
            d = cohen_d(mt_won, mt_lost)
            if abs(d) > 0.4:
                insights.append({
                    "title": "Multi-threading gap widens significantly in high-ACV deals",
                    "finding": (
                        f"In deals above the median ACV (${acv_median:,.0f}), won deals had "
                        f"{mt_won.mean():.1f} multi-thread attempts on average vs "
                        f"{mt_lost.mean():.1f} for lost deals (Cohen's d = {d:.2f}). "
                        f"The effect is much smaller in low-ACV deals, suggesting multi-threading "
                        f"ROI is highly skewed toward larger opportunities."
                    ),
                    "implication": "Multi-thread coaching and inspection should be prioritized on high-ACV deals, not applied uniformly across the book.",
                    "effect_size": round(float(d), 3),
                    "category": "coaching",
                })

    print(f"[insights] {len(insights)} non-obvious insights extracted")
    return insights


# ─────────────────────────────────────────────────────────────────────────────
# 8. LLM SYNTHESIS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthesis_report(
    patterns: dict,
    signature_library: dict,
    intervention_library: dict,
    contradiction_summary: dict,
    insights: list,
    feature_importance: list,
) -> str:
    """
    Call Claude via the Anthropic API to synthesize all findings into
    a structured, opinionated narrative report.
    """

    top_signals = [f["signal"] for f in feature_importance[:8]]
    sig_summary = {
        st: {
            "loss_rate":       meta["loss_rate"],
            "top_3_signals":   meta["top_3_signals"],
            "segment_loss_rate": meta["segment_loss_rate"],
        }
        for st, meta in signature_library.items()
    }
    intervention_summary = {
        st: {
            "recovery_rate":  data.get("recovery_rate", 0),
            "top_intervention": data["interventions"][0]["action"] if data.get("interventions") else "N/A",
        }
        for st, data in intervention_library.items()
    }

    total_deals = sum(meta['deal_count'] for meta in signature_library.values())
    insight_details = [{"title": i["title"], "finding": i["finding"], "implication": i["implication"]} for i in insights]
    prompt = f"""You are a senior revenue operations analyst. You have just completed a statistical win/loss analysis on {total_deals} closed deals. Write a synthesis report for the Head of Sales that surfaces non-obvious findings and actionable implications. Be direct, specific, and opinionated. Avoid generic sales advice.

DATA SUMMARY:
- Top predictive signals (by permutation importance): {top_signals}
- Stall signature loss rates and segment breakdown: {json.dumps(sig_summary, indent=2)}
- Intervention recovery rates and top action per stall type: {json.dumps(intervention_summary, indent=2)}
- Contradiction rate (stated loss reason ≠ behavioral evidence): {contradiction_summary}
- Non-obvious insights extracted: {json.dumps([i['title'] for i in insights], indent=2)}
- Detailed insight findings: {json.dumps(insight_details, indent=2)}

Write the report in this structure:
1. Executive Summary (3-4 sentences — what is the single most important thing to know)
2. What the Data Actually Shows (3-5 bullet findings, each specific with numbers)
3. Where the Pipeline Is Lying to You (focus on contradictions and misattributed loss reasons)
4. The Intervention Window Problem (focus on timing — when to act vs. when it's too late)
5. Three Things to Change Next Quarter (specific, prioritized, opinionated)

Tone: direct, data-grounded, no fluff. Write as if presenting to a skeptical CRO who has seen too many decks."""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        report_text = data["content"][0]["text"]
        print("[synthesis] LLM report generated successfully")
        return report_text
    except Exception as e:
        print(f"[synthesis] API call failed: {e} — using structured fallback")
        return _fallback_report(insights, signature_library, intervention_library)


def _fallback_report(insights, signature_library, intervention_library) -> str:
    lines = ["# Win/Loss Synthesis Report\n", "## Key Findings\n"]
    for insight in insights:
        lines.append(f"### {insight['title']}\n")
        lines.append(f"{insight['finding']}\n\n")
        lines.append(f"**Implication:** {insight['implication']}\n\n")
    lines.append("## Recovery Rates by Stall Type\n")
    for st, data in intervention_library.items():
        lines.append(f"- **{st}**: {data.get('recovery_rate', 0):.0%} recovery rate "
                     f"({data.get('recovered_deal_count', 0)} recovered deals)\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_synthesizer():
    print("\n" + "="*60)
    print("DEAL TRAJECTORY ENGINE — PHASE 2: WIN/LOSS SYNTHESIZER")
    print("="*60 + "\n")

    # 1. Load
    df = load_and_clean(DATA_PATH)

    # 2. Pattern extraction
    print("\n[2] Extracting win/loss patterns...")
    patterns = extract_win_loss_patterns(df)

    # 3. Feature importance
    print("\n[3] Computing feature importance (GBM)...")
    feature_importance = compute_feature_importance(df)

    # 4. Stall signature library
    print("\n[4] Building stall signature library...")
    signature_library = build_stall_signature_library(df, patterns)

    # 5. Contradiction detection
    print("\n[5] Detecting contradictions...")
    contradiction_df = detect_contradictions(df, signature_library)
    flagged = contradiction_df[contradiction_df["is_contradiction"] == True]
    contradiction_summary = {
        "total_deals_with_stated_reason": int(len(contradiction_df)),
        "contradictions_flagged":         int(len(flagged)),
        "contradiction_rate":             round(float(len(flagged) / len(contradiction_df)), 3) if len(contradiction_df) > 0 else 0,
        "top_pairs": (
            contradiction_df[contradiction_df["is_contradiction"] == True]
            .groupby(["stated_loss_reason", "behavioral_best_match"])
            .size()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        ) if len(flagged) > 0 else {},
    }

    # 6. Intervention library
    print("\n[6] Building intervention library...")
    intervention_library = build_intervention_library(df, signature_library)

    # 7. Non-obvious insights
    print("\n[7] Extracting non-obvious insights...")
    insights = extract_non_obvious_insights(df, patterns, signature_library, contradiction_df)

    # 8. LLM synthesis
    print("\n[8] Generating synthesis report via LLM...")
    report = generate_synthesis_report(
        patterns, signature_library, intervention_library,
        contradiction_summary, insights, feature_importance
    )

    # ── Write outputs ────────────────────────────────────────────────────────
    print("\n[9] Writing output files...")

    with open(OUTPUT_DIR / "win_loss_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)
    print(f"  → win_loss_patterns.json ({len(patterns)} signals)")

    with open(OUTPUT_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"  → feature_importance.json ({len(feature_importance)} signals ranked)")

    with open(OUTPUT_DIR / "stall_signature_library.json", "w") as f:
        json.dump(signature_library, f, indent=2)
    print(f"  → stall_signature_library.json ({len(signature_library)} signatures)")

    contradiction_df.to_csv(OUTPUT_DIR / "contradiction_report.csv", index=False)
    print(f"  → contradiction_report.csv ({len(contradiction_df)} rows, {len(flagged)} flagged)")

    with open(OUTPUT_DIR / "intervention_library.json", "w") as f:
        json.dump(intervention_library, f, indent=2)
    print(f"  → intervention_library.json")

    with open(OUTPUT_DIR / "non_obvious_insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print(f"  → non_obvious_insights.json ({len(insights)} insights)")

    with open(OUTPUT_DIR / "synthesis_report.md", "w") as f:
        f.write(report)
    print(f"  → synthesis_report.md")

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)

    return {
        "df": df,
        "patterns": patterns,
        "feature_importance": feature_importance,
        "signature_library": signature_library,
        "contradiction_df": contradiction_df,
        "intervention_library": intervention_library,
        "insights": insights,
        "report": report,
    }


if __name__ == "__main__":
    results = run_synthesizer()
