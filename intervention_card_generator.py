"""
Deal Trajectory Engine — Phase 4: Intervention Card Generator
=============================================================
Takes the scored pipeline and intervention queue from Phase 3 and produces:

  1. Per-deal intervention cards (markdown) — LLM-personalized, deal-specific
  2. Daily manager brief — top 10 priority actions for today
  3. Bundled JSON payload — consumed by the Phase 4 React dashboard

The LLM call takes: deal context, stall pattern, signal evidence, historical
comp deals, rep notes, contradiction flag → personalized first-move draft.
"""

import json
import re
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR    = Path(__file__).parent
QUEUE_PATH    = SCRIPT_DIR / "intervention_queue.json"
SCORED_PATH   = SCRIPT_DIR / "scored_pipeline.csv"
SUMMARY_PATH  = SCRIPT_DIR / "predictor_summary.json"
OUTPUT_DIR    = SCRIPT_DIR

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

STALL_COLORS = {
    "Champion Collapse":        "#E8794A",
    "Competitor Displacement":  "#C44B6B",
    "Ghost Stall":              "#7B6FBE",
    "Exec Vacuum":              "#3B82B8",
}

TRAJECTORY_COLORS = {
    "critical": "#DC2626",
    "at-risk":  "#D97706",
    "watch":    "#2563EB",
    "healthy":  "#16A34A",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM-PERSONALIZED FIRST-MOVE DRAFT
# ─────────────────────────────────────────────────────────────────────────────

def generate_first_move(card: dict) -> str:
    """
    Call Claude to write a personalized first-move message draft for the rep.
    Falls back to a template-based draft if the API call fails.
    """
    company      = card.get("company_name", "the prospect")
    rep_name     = card.get("rep_name", "the rep").split()[0]
    stall_type   = card.get("dominant_stall_type", "")
    evidence     = card.get("evidence", [])
    stage        = card.get("current_stage", "")
    acv          = card.get("acv_usd", 0)
    segment      = card.get("segment", "")
    top_interv   = card.get("top_intervention", {}) or {}
    contradiction = card.get("note_contradiction")

    evidence_text = "\n".join(f"  - {e}" for e in evidence[:3])
    contradiction_note = (
        f"\nIMPORTANT: {contradiction['flag']}" if contradiction else ""
    )

    prompt = f"""You are a senior sales coach drafting a specific, actionable first move for a rep.

DEAL CONTEXT:
- Company: {company}
- Rep: {rep_name}
- Segment: {segment}, ACV: ${acv:,.0f}
- Stage: {stage}
- Stall Pattern: {stall_type}
- Behavioral evidence:
{evidence_text}{contradiction_note}

RECOMMENDED ACTION: {top_interv.get('action', '')}
RATIONALE: {top_interv.get('rationale', '')}

Write a specific first move for {rep_name} to take TODAY. Include:
1. One concrete action (call, email, or internal move) — be specific about the WHO and WHAT
2. A draft opening line or email subject if outreach is involved — something that doesn't sound like a follow-up
3. One thing NOT to do (the common mistake for this pattern)

Keep it under 120 words. Be direct. No fluff. Write as if you're a colleague, not a consultant."""

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception:
        return _template_first_move(stall_type, company, rep_name, top_interv)


def _template_first_move(stall_type: str, company: str, rep_name: str, top_interv: dict) -> str:
    templates = {
        "Champion Collapse": (
            f"**{rep_name} — action required today.**\n\n"
            f"Your champion thread at {company} is collapsing. Don't send another follow-up to the same contact.\n\n"
            f"**Do this:** Map the org and find one additional stakeholder — their manager, a peer on the team, "
            f"or whoever owns budget. Reach out directly with: *'I wanted to introduce myself separately — "
            f"I know [champion] has been our main contact but I'd love 15 min to hear your perspective.'*\n\n"
            f"**Don't:** Ask your champion to make the intro. If they could, they would have by now."
        ),
        "Competitor Displacement": (
            f"**{rep_name} — competitive threat at {company}.**\n\n"
            f"The behavioral data shows a competitor is gaining ground. Don't discount yet.\n\n"
            f"**Do this:** Schedule a working session (not a demo) focused on their specific technical environment. "
            f"Open with: *'I want to spend 45 min going deeper on [specific use case] — not slides, "
            f"just your data and our stack together.'*\n\n"
            f"**Don't:** Respond to price pressure by discounting. Recovered deals competed on depth, not price."
        ),
        "Ghost Stall": (
            f"**{rep_name} — {company} has gone dark.**\n\n"
            f"More emails won't work. Change the medium before you change the message.\n\n"
            f"**Do this:** Call or LinkedIn message today with: *'Not chasing you — just want to know "
            f"if the [original problem] is still on the table or if priorities shifted. "
            f"One sentence is fine.'*\n\n"
            f"**Don't:** Send another 'just checking in' email. You've already sent too many."
        ),
        "Exec Vacuum": (
            f"**{rep_name} — no economic buyer visibility at {company}.**\n\n"
            f"Your champion can't close this alone. You need a name above them this week.\n\n"
            f"**Do this:** Ask your champion directly: *'Who else needs to say yes for this to happen? "
            f"Can you help me get 20 min with them before end of month?'*\n\n"
            f"**Don't:** Wait for the champion to bring it up. They won't — it's uncomfortable for them too."
        ),
    }
    return templates.get(stall_type, top_interv.get("action", "Review deal and define next step."))


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD INDIVIDUAL MARKDOWN CARDS
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown_card(card: dict, first_move: str) -> str:
    trajectory  = card.get("trajectory", "").upper()
    stall       = card.get("dominant_stall_type", "")
    score_pct   = round(card.get("stall_score", 0) * 100)
    evidence    = card.get("evidence", [])
    hist        = card.get("historical_context", "")
    contradiction = card.get("note_contradiction")
    all_interventions = card.get("top_intervention")

    evidence_md = "\n".join(f"- {e}" for e in evidence)
    contradiction_md = (
        f"\n> ⚠️ **Contradiction detected:** {contradiction['flag']}\n"
        if contradiction else ""
    )

    return f"""# Intervention Card — {card.get('company_name', 'Unknown')}

**Deal ID:** {card.get('deal_id')}  
**Rep:** {card.get('rep_name')}  
**Segment:** {card.get('segment')} | **ACV:** ${card.get('acv_usd', 0):,.0f}  
**Stage:** {card.get('current_stage')} | **Urgency:** {card.get('urgency_stage', '').upper()}

---

## 🚨 Trajectory: {trajectory} — {stall}

**Signal coverage:** {card.get('signal_coverage')}  
**Stall score:** {score_pct}%  
{card.get('stall_description', '')}

---

## Evidence

{evidence_md}
{contradiction_md}
---

## Historical Context

{hist}

---

## First Move — Do This Today

{first_move}

---

## Full Intervention Playbook

{_format_intervention(all_interventions)}

---
*Generated: {card.get('generated_at')} | Priority score: {card.get('priority_score')}*
"""


def _format_intervention(interv: dict | None) -> str:
    if not interv:
        return "_No intervention data available._"
    return (
        f"**Priority {interv.get('priority', 1)} action:** {interv.get('action', '')}\n\n"
        f"**Why this works:** {interv.get('rationale', '')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. DAILY MANAGER BRIEF
# ─────────────────────────────────────────────────────────────────────────────

def build_manager_brief(queue: list, summary: dict, first_moves: dict) -> str:
    today        = datetime.now().strftime("%A, %B %d %Y")
    total        = summary.get("total_pipeline_deals", 0)
    flagged      = summary.get("flagged_deal_count", 0)
    acv_at_risk  = summary.get("acv_at_risk", 0)
    traj         = summary.get("trajectory_distribution", {})
    contradicts  = summary.get("contradiction_flags", 0)
    stall_dist   = summary.get("stall_type_distribution", {})
    top5         = queue[:5]

    # Stall distribution bar
    stall_lines = "\n".join(
        f"| {st:<30} | {count:>3} deals |"
        for st, count in sorted(stall_dist.items(), key=lambda x: -x[1])
    )

    # Top 10 action items
    action_items = []
    for i, card in enumerate(queue[:10], 1):
        fm = first_moves.get(card["deal_id"], "")
        # Extract first sentence of first move as the action line
        first_line = fm.split("\n\n")[0].replace("**", "").strip()
        action_items.append(
            f"{i}. **{card['company_name']}** ({card['segment']}, ${card['acv_usd']:,.0f}) "
            f"— [{card['trajectory'].upper()}] {card['dominant_stall_type']}\n"
            f"   → {card['evidence'][0] if card['evidence'] else 'Review signals'}\n"
            f"   → **Action:** {card['top_intervention']['action'][:100] if card.get('top_intervention') else 'See card'}..."
        )

    action_md = "\n\n".join(action_items)

    # Reps with multiple flags
    rep_flags   = summary.get("reps_with_most_flags", {})
    rep_lines   = "\n".join(
        f"- {rep}: {count} flagged deals"
        for rep, count in list(rep_flags.items())[:5]
    )

    return f"""# Deal Trajectory Engine — Daily Manager Brief
**{today}**

---

## Pipeline Health Snapshot

| Metric | Value |
|--------|-------|
| Deals scored | {total} |
| Flagged (at-risk + critical) | {flagged} ({round(flagged/total*100, 1)}%) |
| ACV at risk | ${acv_at_risk:,.0f} |
| Critical | {traj.get('critical', 0)} deals |
| At-Risk | {traj.get('at-risk', 0)} deals |
| Watch | {traj.get('watch', 0)} deals |
| Contradiction flags | {contradicts} (rep notes ≠ behavioral signals) |

---

## Stall Pattern Breakdown

| Pattern | Count |
|---------|-------|
{stall_lines}

---

## Top 10 Priority Actions Today

{action_md}

---

## Reps with Most Flagged Deals

{rep_lines}

---

## What to Watch

- **{contradicts} deals** have rep notes that contradict the behavioral evidence.
  These are the highest-risk coaching conversations — the rep thinks the deal is
  one thing, the data says it's another.

- **Ghost Stall** accounts for {stall_dist.get('Ghost Stall', 0)} flagged deals.
  These are the easiest to ignore and the easiest to lose. Set a hard rule:
  any deal with >12 days of silence gets a channel-change attempt today, not next week.

- **Exec Vacuum** is the pattern most likely to result in a "no decision" outcome.
  If a deal is in Negotiation or Proposal Sent with no economic buyer meeting logged,
  that deal has a low close probability regardless of what the rep says.

---
*Generated by Deal Trajectory Engine v1.0 | Data as of {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD DASHBOARD PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_nans(obj):
    """Recursively replace float NaN with None so strict JSON parsers accept the output."""
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_dashboard_payload(queue: list, scored_df: pd.DataFrame,
                             summary: dict, first_moves: dict) -> dict:
    """
    Serialize everything into a single JSON blob consumed by the React dashboard.
    Keeps the UI stateless — all data is embedded at build time.
    """
    # Enrich each card with its LLM first move
    enriched_cards = []
    for card in queue:
        enriched = {**card, "first_move": first_moves.get(card["deal_id"], "")}
        # Clean numpy types for JSON serialization
        for k, v in enriched.items():
            if isinstance(v, (np.integer, np.int64)):
                enriched[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                enriched[k] = float(v)
        enriched_cards.append(enriched)

    # Pipeline-wide signal stats for overview charts
    signal_stats = {
        "trajectory_distribution": summary.get("trajectory_distribution", {}),
        "stall_type_distribution": summary.get("stall_type_distribution", {}),
        "acv_at_risk":             summary.get("acv_at_risk", 0),
        "acv_total":               summary.get("total_acv_scored", 0),
        "total_deals":             summary.get("total_pipeline_deals", 0),
        "flagged_deals":           summary.get("flagged_deal_count", 0),
        "contradiction_flags":     summary.get("contradiction_flags", 0),
        "top_5_priority_deals":    summary.get("top_5_priority_deals", []),
        "reps_with_most_flags":    summary.get("reps_with_most_flags", {}),
    }

    # Segment breakdown from scored df
    seg_traj = (
        scored_df.groupby(["segment", "trajectory"])
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )

    return {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_stats":   signal_stats,
        "segment_breakdown": seg_traj,
        "intervention_queue": enriched_cards,
        "stall_colors":   STALL_COLORS,
        "trajectory_colors": TRAJECTORY_COLORS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_card_generator():
    print("\n" + "="*60)
    print("DEAL TRAJECTORY ENGINE — PHASE 4: CARD GENERATOR")
    print("="*60 + "\n")

    # Load
    print("[1] Loading Phase 3 outputs...")
    with open(QUEUE_PATH) as f:
        queue = json.load(f)
    scored_df = pd.read_csv(SCORED_PATH, low_memory=False)
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    print(f"  → {len(queue)} intervention cards to process")

    # Generate first moves (LLM or template fallback)
    print("\n[2] Generating personalized first moves...")
    first_moves = {}
    for i, card in enumerate(queue):
        deal_id = card["deal_id"]
        fm = generate_first_move(card)
        first_moves[deal_id] = fm
        if (i + 1) % 10 == 0:
            print(f"  → {i+1}/{len(queue)} cards processed")
    print(f"  → All {len(queue)} first moves generated")

    # Build markdown cards for top 20
    print("\n[3] Building markdown cards (top 20 by priority)...")
    cards_dir = OUTPUT_DIR / "cards"
    cards_dir.mkdir(exist_ok=True)
    for card in queue[:20]:
        md = build_markdown_card(card, first_moves[card["deal_id"]])
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", card["deal_id"])
        with open(cards_dir / f"{safe_name}.md", "w", encoding="utf-8") as f:
            f.write(md)
    print(f"  → {min(20, len(queue))} markdown cards written to outputs/cards/")

    # Build manager brief
    print("\n[4] Building daily manager brief...")
    brief = build_manager_brief(queue, summary, first_moves)
    with open(OUTPUT_DIR / "manager_brief.md", "w", encoding="utf-8") as f:
        f.write(brief)
    print("  → manager_brief.md written")

    # Build dashboard payload
    print("\n[5] Building dashboard payload...")
    payload = build_dashboard_payload(queue, scored_df, summary, first_moves)
    with open(OUTPUT_DIR / "dashboard_payload.json", "w", encoding="utf-8") as f:
        # allow_nan=False + pre-sanitize ensures the JSON parses cleanly in
        # build_dashboard.js (which uses strict JSON.parse).
        json.dump(_sanitize_nans(payload), f, indent=2, default=str, allow_nan=False)
    print(f"  → dashboard_payload.json written ({len(payload['intervention_queue'])} cards)")

    print("\n" + "="*60)
    print("PHASE 4 COMPLETE — READY FOR DASHBOARD")
    print("="*60)

    return payload


if __name__ == "__main__":
    payload = run_card_generator()
