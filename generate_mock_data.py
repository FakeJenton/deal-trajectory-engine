"""
Deal Trajectory Engine — Mock Data Generator
Generates two datasets:
  1. historical_deals.csv  — closed won/lost deals used to extract stall signatures
  2. active_pipeline.csv   — live deals to score against those signatures

Data is deliberately imperfect: missing fields, duplicate rows, inconsistent
formatting, free-text contradictions, and conflicting stated vs. behavioral signals.
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# REFERENCE TABLES
# ─────────────────────────────────────────────

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = [
    "Financial Services", "Healthcare", "Retail", "Technology",
    "Manufacturing", "Logistics", "Insurance", "Media", "Energy", "Education"
]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest", "EMEA", "APAC"]
LEAD_SOURCES = [
    "Outbound SDR", "Inbound - Website", "Partner Referral", "Event / Conference",
    "Executive Referral", "LinkedIn Outbound", "Cold Email Sequence", "PLG Expansion"
]
COMPETITORS = [
    "Snowflake", "Databricks", "Fivetran", "dbt Labs", "Starburst",
    "Monte Carlo", "Atlan", "Alation", "Collibra", "None / Greenfield"
]
PRODUCT_LINES = ["Core Platform", "Analytics Suite", "Governance Module", "Full Bundle"]
CONTRACT_TYPES = ["Annual", "Multi-Year", "Monthly", "Pilot"]
LOSS_REASONS = [
    "Price / Budget", "Chose Competitor", "No Decision / Stalled",
    "Wrong Timing", "Missing Feature", "Champion Left", "Security / Compliance Failed"
]
CHAMPION_TITLES = [
    "Data Engineer", "Senior Data Engineer", "Lead Data Engineer",
    "Analytics Engineer", "Data Platform Lead", "Head of Data Engineering",
    "Director of Data", "VP of Data", "Chief Data Officer",
    "Analytics Manager", "BI Lead", "Data Architect"
]
ECONOMIC_BUYER_TITLES = [
    "VP Engineering", "CTO", "CFO", "COO", "SVP Technology",
    "Director of IT", "Head of Product", "Chief Data Officer", "VP Operations"
]
SENIORITY_MAP = {
    "Data Engineer": "IC", "Senior Data Engineer": "IC", "Lead Data Engineer": "IC",
    "Analytics Engineer": "IC", "Data Platform Lead": "Manager",
    "Head of Data Engineering": "Manager", "Director of Data": "Director",
    "VP of Data": "VP", "Chief Data Officer": "C-Suite",
    "Analytics Manager": "Manager", "BI Lead": "Manager", "Data Architect": "IC"
}
STAGE_NAMES = [
    "Prospecting", "Discovery", "Technical Validation",
    "Proposal Sent", "Negotiation", "Closed Won", "Closed Lost"
]

# Stall signatures — behavioral thresholds that define each pattern
# Used later in Phase 2; defined here so generation is consistent
STALL_SIGNATURES = {
    "Champion Collapse": {
        "email_response_latency_delta": ">= 5",   # days increase
        "unique_stakeholders_midpoint": "<= 1",
        "days_since_multithread": ">= 18",
        "champion_engagement_score_delta": "<= -20",
    },
    "Competitor Displacement": {
        "competitive_mentions_count": ">= 3",
        "pricing_objection_raised": True,
        "days_since_last_activity": ">= 10",
        "stage_regression_count": ">= 1",
    },
    "Ghost Stall": {
        "days_since_last_rep_touch": ">= 12",
        "email_response_rate": "<= 0.25",
        "meeting_count_last_30d": "== 0",
        "days_in_current_stage": ">= 25",
    },
    "Exec Vacuum": {
        "executive_sponsor_engaged": False,
        "days_to_first_exec_engagement": ">= 50",
        "economic_buyer_meetings": "== 0",
        "champion_seniority": "IC",
    }
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def jitter(val, pct=0.15):
    """Add proportional noise to a numeric value."""
    return round(val * (1 + random.uniform(-pct, pct)), 2)

def maybe_null(val, null_rate=0.08):
    """Randomly blank out a value to simulate missing CRM data."""
    return None if random.random() < null_rate else val

def dirty_company_name(name):
    """Introduce realistic formatting inconsistencies."""
    variants = [
        name,
        name.upper(),
        name.lower(),
        name + ", Inc.",
        name + " Inc",
        name + " LLC",
        name.replace(" ", ""),
        "The " + name,
    ]
    return random.choice(variants)

def messy_date(dt):
    """Return date in one of several inconsistent formats."""
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y"]
    return dt.strftime(random.choice(formats)) if dt else None

def random_date(start_days_ago=730, end_days_ago=30):
    days = random.randint(end_days_ago, start_days_ago)
    return datetime.today() - timedelta(days=days)

def rep_notes_template(outcome, stall_type=None, competitor=None):
    won_notes = [
        "Strong champion, got exec in early. Smooth close.",
        "Technical win — their data architect loved the architecture demo.",
        "Competitive deal vs {c}, but we had the better integration story.",
        "Multi-threaded well. Finance and Eng both bought in.",
        "Fast cycle. Champion had budget approved before we even started.",
        "Close was delayed 2 weeks on legal but closed strong.",
    ]
    lost_notes_real = {
        "Champion Collapse": [
            "Our champion got moved to a different team mid-deal. Lost access.",
            "New data lead came in and had a prior relationship with Snowflake.",
            "Champion left the company. Couldn't rebuild the relationship in time.",
        ],
        "Competitor Displacement": [
            "They went with Databricks. Had deeper existing footprint.",
            "Snowflake offered a massive discount at the last minute. We couldn't match.",
            "Lost to Fivetran on connector breadth. They had 200+ native connectors.",
        ],
        "Ghost Stall": [
            "Just went dark. Last email 3 weeks ago, no response.",
            "Prospect stopped engaging after the proposal. No explanation given.",
            "Multiple follow-ups with no reply. Probably went internal.",
        ],
        "Exec Vacuum": [
            "Never got above the data team. Budget decision was above champion's level.",
            "No exec sponsor. Deal died when Q4 budgets froze.",
            "Champion was IC level, couldn't drive the purchase alone.",
        ],
    }
    stated_loss_contradictions = {
        # behavioral pattern → stated reason that contradicts it
        "Champion Collapse": "Price / Budget",
        "Competitor Displacement": "No Decision / Stalled",
        "Ghost Stall": "Missing Feature",
        "Exec Vacuum": "Price / Budget",
    }

    if outcome == "Won":
        note = random.choice(won_notes).replace("{c}", competitor or "the incumbent")
    elif stall_type and random.random() < 0.45:
        # Real note (matches behavior) ~55% of the time, contradicting stated reason ~45%
        note = random.choice(lost_notes_real.get(stall_type, ["No notes logged."]))
    else:
        note = f"Lost — {random.choice(LOSS_REASONS).lower()}. No additional detail."

    # Introduce typos and informal formatting in ~20% of notes
    if random.random() < 0.20:
        note = note.replace(".", "..").replace(" the ", " teh ").lower()

    return note


# ─────────────────────────────────────────────
# CORE RECORD BUILDER
# ─────────────────────────────────────────────

def build_deal(deal_id, closed=True, outcome=None, force_stall=None):
    """
    Build one deal record.
    closed=True  → historical deal (has outcome)
    closed=False → active pipeline deal (no outcome yet)
    force_stall  → inject a specific stall signature into behavioral fields
    """
    segment = random.choice(SEGMENTS)
    industry = random.choice(INDUSTRIES)
    region = random.choice(REGIONS)
    competitor = random.choice(COMPETITORS)
    product = random.choice(PRODUCT_LINES)
    contract = random.choice(CONTRACT_TYPES)
    lead_source = random.choice(LEAD_SOURCES)
    rep_name = fake.name()
    rep_id = f"REP-{random.randint(100,999)}"

    champion_title = random.choice(CHAMPION_TITLES)
    champion_seniority = SENIORITY_MAP[champion_title]
    economic_buyer_title = random.choice(ECONOMIC_BUYER_TITLES)

    # ACV by segment
    acv_ranges = {"SMB": (8000, 35000), "Mid-Market": (35000, 150000), "Enterprise": (150000, 800000)}
    acv = round(random.uniform(*acv_ranges[segment]), -2)

    discount_pct = round(random.uniform(0, 0.30), 2) if random.random() > 0.4 else 0.0
    net_acv = round(acv * (1 - discount_pct), -2)

    cycle_days_base = {"SMB": 30, "Mid-Market": 75, "Enterprise": 160}[segment]
    sales_cycle_days = int(jitter(cycle_days_base, pct=0.4))

    close_date = random_date(730, 30) if closed else None
    created_date = (close_date - timedelta(days=sales_cycle_days)) if close_date else (datetime.today() - timedelta(days=random.randint(10, 90)))

    # ── Behavioral signals ──────────────────────────────

    # Defaults for a healthy deal
    email_resp_latency_start = round(random.uniform(0.5, 2.5), 1)
    email_resp_latency_end = round(random.uniform(0.5, 3.0), 1)
    unique_stakeholders = random.randint(2, 8)
    stakeholders_at_midpoint = max(1, unique_stakeholders - random.randint(0, 2))
    days_since_multithread = random.randint(1, 15)
    days_since_last_activity = random.randint(0, 10)
    days_since_last_rep_touch = random.randint(0, 8)
    call_frequency = round(random.uniform(0.5, 3.0), 1)
    total_calls = random.randint(3, 20)
    total_emails_sent = random.randint(10, 60)
    total_emails_received = random.randint(5, total_emails_sent)
    email_response_rate = round(total_emails_received / total_emails_sent, 2)
    meeting_count = random.randint(2, 12)
    meeting_count_last_30d = random.randint(1, 5)
    demo_count = random.randint(1, 4)
    proposal_sent = random.choice([True, False])
    days_to_proposal = random.randint(10, 45) if proposal_sent else None
    days_in_current_stage = random.randint(3, 20)
    stage_regression_count = random.randint(0, 1)
    stage_change_count = random.randint(2, 6)
    champion_engagement_score = random.randint(55, 95)
    champion_engagement_score_delta = random.randint(-10, 15)
    competitive_mentions_count = random.randint(0, 2)
    pricing_objection_raised = random.choice([True, False])
    technical_validation_complete = random.choice([True, False])
    security_review_complete = random.choice([True, False])
    legal_review_started = random.choice([True, False])
    executive_sponsor_engaged = random.choice([True, True, False])
    days_to_first_exec_engagement = random.randint(5, 35) if executive_sponsor_engaged else None
    economic_buyer_meetings = random.randint(1, 5) if executive_sponsor_engaged else 0
    nps_score = maybe_null(random.randint(6, 10))
    intent_score = maybe_null(random.randint(50, 95))
    website_visits_last_30d = random.randint(2, 25)
    product_trial_active = random.choice([True, False])
    trial_days_active = random.randint(5, 30) if product_trial_active else None
    open_support_tickets = random.randint(0, 2)

    # ── Override behavioral signals based on stall signature ──

    stall_type = force_stall
    if stall_type == "Champion Collapse":
        email_resp_latency_end = jitter(email_resp_latency_start + 6.5)
        stakeholders_at_midpoint = 1
        days_since_multithread = random.randint(18, 40)
        champion_engagement_score_delta = random.randint(-35, -20)
        champion_engagement_score = random.randint(20, 50)
        executive_sponsor_engaged = False
        days_to_first_exec_engagement = None
    elif stall_type == "Competitor Displacement":
        competitive_mentions_count = random.randint(3, 7)
        pricing_objection_raised = True
        days_since_last_activity = random.randint(10, 20)
        stage_regression_count = random.randint(1, 3)
        discount_pct = round(random.uniform(0.20, 0.35), 2)
    elif stall_type == "Ghost Stall":
        days_since_last_rep_touch = random.randint(12, 35)
        email_response_rate = round(random.uniform(0.05, 0.25), 2)
        meeting_count_last_30d = 0
        days_in_current_stage = random.randint(25, 60)
        days_since_last_activity = random.randint(12, 40)
        total_emails_received = max(1, int(total_emails_sent * email_response_rate))
    elif stall_type == "Exec Vacuum":
        executive_sponsor_engaged = False
        days_to_first_exec_engagement = None
        economic_buyer_meetings = 0
        champion_seniority = "IC"
        champion_title = random.choice(["Data Engineer", "Senior Data Engineer", "Analytics Engineer"])

    # ── Outcome and labels ──────────────────────────────────

    if closed:
        if outcome is None:
            # Stalled deals lose more often
            outcome = "Lost" if (stall_type or random.random() < 0.45) else "Won"

        if outcome == "Won":
            stage = "Closed Won"
            stated_loss_reason = None
        else:
            # Stated reason may contradict behavioral pattern
            contradiction_map = {
                "Champion Collapse": "Price / Budget",
                "Competitor Displacement": "No Decision / Stalled",
                "Ghost Stall": "Missing Feature",
                "Exec Vacuum": "Price / Budget",
            }
            if stall_type and random.random() < 0.45:
                stated_loss_reason = contradiction_map.get(stall_type, random.choice(LOSS_REASONS))
            else:
                behavioral_to_reason = {
                    "Champion Collapse": "Champion Left",
                    "Competitor Displacement": "Chose Competitor",
                    "Ghost Stall": "No Decision / Stalled",
                    "Exec Vacuum": "No Decision / Stalled",
                    None: random.choice(LOSS_REASONS),
                }
                stated_loss_reason = behavioral_to_reason.get(stall_type, random.choice(LOSS_REASONS))
            stage = "Closed Lost"
    else:
        outcome = None
        stated_loss_reason = None
        stage = random.choice(STAGE_NAMES[:5])

    rep_notes = rep_notes_template(outcome or "Active", stall_type, competitor)

    # ── Last email subject line (messy free text) ───────────────

    email_subjects = [
        "Re: Follow up from our call",
        "Quick question on the proposal",
        "RE: RE: RE: Next steps??",
        "Checking in",
        "fwd: technical review docs",
        "Q4 pricing — URGENT",
        "[External] Re: Tiger Data POC",
        "following up again...",
        "Introduction: {name} <> Tiger Data",
        None,
    ]
    last_email_subject = maybe_null(random.choice(email_subjects), null_rate=0.12)
    if last_email_subject and "{name}" in last_email_subject:
        last_email_subject = last_email_subject.replace("{name}", fake.last_name())

    # ── Assemble record ─────────────────────────────────────────

    record = {
        # ── Identifiers ──
        "deal_id": deal_id,
        "company_name": maybe_null(dirty_company_name(fake.company()), null_rate=0.01),
        "crm_account_id": f"ACC-{random.randint(10000,99999)}",
        "opportunity_name": maybe_null(f"{fake.company()} - {product} {contract}"),

        # ── Deal metadata ──
        "segment": segment,
        "industry": industry,
        "region": region,
        "rep_name": rep_name,
        "rep_id": rep_id,
        "account_owner": maybe_null(fake.name(), null_rate=0.1),
        "sdr_name": maybe_null(fake.name(), null_rate=0.2),
        "product_line": product,
        "contract_type": contract,
        "lead_source": lead_source,
        "competitor_primary": competitor,
        "competitor_secondary": maybe_null(random.choice(COMPETITORS), null_rate=0.6),

        # ── Financials ──
        "acv_usd": acv,
        "net_acv_usd": net_acv,
        "tcv_usd": maybe_null(round(net_acv * (1 if contract == "Annual" else random.uniform(2, 3)), -2)),
        "discount_pct": discount_pct,
        "budget_confirmed": maybe_null(random.choice([True, False]), null_rate=0.15),
        "budget_range_stated": maybe_null(random.choice(["<$50k", "$50k–$150k", "$150k–$500k", ">$500k"]), null_rate=0.25),

        # ── Timeline ──
        "created_date": messy_date(created_date),
        "close_date": messy_date(close_date),
        "sales_cycle_days": sales_cycle_days if closed else None,
        "days_in_current_stage": days_in_current_stage,
        "stage_change_count": stage_change_count,
        "stage_regression_count": stage_regression_count,
        "days_to_proposal": days_to_proposal,
        "days_to_first_exec_engagement": days_to_first_exec_engagement,
        "current_stage": stage,

        # ── Contacts & stakeholders ──
        "champion_name": maybe_null(fake.name(), null_rate=0.05),
        "champion_title": champion_title,
        "champion_seniority": champion_seniority,
        "champion_tenure_at_company_yrs": maybe_null(round(random.uniform(0.5, 12), 1), null_rate=0.2),
        "economic_buyer_name": maybe_null(fake.name(), null_rate=0.15),
        "economic_buyer_title": economic_buyer_title,
        "executive_sponsor_engaged": executive_sponsor_engaged,
        "economic_buyer_meetings": economic_buyer_meetings,
        "unique_stakeholders_total": unique_stakeholders,
        "unique_stakeholders_at_midpoint": stakeholders_at_midpoint,
        "it_security_contact_identified": maybe_null(random.choice([True, False]), null_rate=0.2),
        "legal_contact_identified": maybe_null(random.choice([True, False]), null_rate=0.2),
        "finance_contact_identified": maybe_null(random.choice([True, False]), null_rate=0.2),

        # ── Email & communication signals ──
        "email_response_latency_start_days": email_resp_latency_start,
        "email_response_latency_end_days": email_resp_latency_end,
        "email_response_latency_delta_days": round(email_resp_latency_end - email_resp_latency_start, 2),
        "email_response_rate": email_response_rate,
        "total_emails_sent": total_emails_sent,
        "total_emails_received": total_emails_received,
        "days_since_last_rep_touch": days_since_last_rep_touch,
        "days_since_last_activity": days_since_last_activity,
        "last_email_subject": last_email_subject,

        # ── Meeting & call signals ──
        "total_calls": total_calls,
        "call_frequency_per_week": call_frequency,
        "meeting_count_total": meeting_count,
        "meeting_count_last_30d": meeting_count_last_30d,
        "demo_count": demo_count,
        "last_meeting_type": maybe_null(random.choice(["Discovery", "Demo", "Technical Review", "Pricing", "Executive QBR", "POC Review"]), null_rate=0.1),

        # ── Multi-thread / engagement breadth ──
        "days_since_multithread_touch": days_since_multithread,
        "multithread_attempt_count": random.randint(0, 5),
        "champion_engagement_score": champion_engagement_score,
        "champion_engagement_score_delta": champion_engagement_score_delta,
        "stakeholder_breadth_trend": maybe_null(random.choice(["Expanding", "Stable", "Contracting"]), null_rate=0.1),

        # ── Competitive & objection signals ──
        "competitive_mentions_count": competitive_mentions_count,
        "pricing_objection_raised": pricing_objection_raised,
        "pricing_objection_count": random.randint(1, 4) if pricing_objection_raised else 0,
        "feature_gap_flagged": maybe_null(random.choice([True, False]), null_rate=0.1),
        "security_objection_raised": maybe_null(random.choice([True, False]), null_rate=0.1),

        # ── Validation & process signals ──
        "proposal_sent": proposal_sent,
        "technical_validation_complete": technical_validation_complete,
        "security_review_complete": security_review_complete,
        "legal_review_started": legal_review_started,
        "legal_review_complete": maybe_null(random.choice([True, False]), null_rate=0.3),
        "procurement_engaged": maybe_null(random.choice([True, False]), null_rate=0.2),
        "msa_redlines_received": maybe_null(random.choice([True, False]), null_rate=0.3),
        "sow_sent": maybe_null(random.choice([True, False]), null_rate=0.2),

        # ── Product / trial signals ──
        "product_trial_active": product_trial_active,
        "trial_days_active": trial_days_active,
        "trial_feature_adoption_score": maybe_null(random.randint(20, 100), null_rate=0.4),
        "integration_poc_complete": maybe_null(random.choice([True, False]), null_rate=0.2),
        "open_support_tickets": open_support_tickets,

        # ── Intent & 3rd-party signals ──
        "intent_score": intent_score,
        "website_visits_last_30d": website_visits_last_30d,
        "g2_profile_viewed": maybe_null(random.choice([True, False]), null_rate=0.3),
        "content_downloads_count": maybe_null(random.randint(0, 8), null_rate=0.3),
        "linkedin_engagement_score": maybe_null(random.randint(0, 100), null_rate=0.35),

        # ── Free text / qualitative ──
        "rep_notes": rep_notes,
        "last_call_notes_snippet": maybe_null(fake.sentence(nb_words=14), null_rate=0.25),
        "nps_score": nps_score,
        "forecast_category": maybe_null(random.choice(["Commit", "Best Case", "Pipeline", "Omitted"]), null_rate=0.1),
        "next_step_defined": maybe_null(random.choice([True, False]), null_rate=0.1),
        "next_step_due_date": maybe_null(messy_date(datetime.today() + timedelta(days=random.randint(1, 30))), null_rate=0.2),
        "mutual_action_plan_shared": maybe_null(random.choice([True, False]), null_rate=0.2),

        # ── Outcome fields (historical only) ──
        "outcome": outcome,
        "stated_loss_reason": stated_loss_reason,
        "injected_stall_signature": stall_type,  # ground truth label for validation
    }

    return record


# ─────────────────────────────────────────────
# DATASET 1: HISTORICAL CLOSED DEALS
# ─────────────────────────────────────────────

def generate_historical_deals(n=150):
    records = []
    deal_counter = 1000

    # Distribute stall types across lost deals
    stall_types = list(STALL_SIGNATURES.keys())
    stall_distribution = []
    for _ in range(n):
        if random.random() < 0.55:  # ~55% lost
            stall_distribution.append(random.choice(stall_types))
        else:
            stall_distribution.append(None)  # Won deal — no stall

    for i in range(n):
        stall = stall_distribution[i]
        outcome = "Lost" if stall else ("Won" if random.random() > 0.3 else "Lost")
        deal_id = f"DL-{deal_counter + i}"
        rec = build_deal(deal_id, closed=True, outcome=outcome, force_stall=stall)
        records.append(rec)

    df = pd.DataFrame(records)

    # ── Introduce deliberate data quality issues ──

    # 1. Duplicate ~3% of rows with minor field variations
    dup_indices = random.sample(range(len(df)), k=int(len(df) * 0.03))
    dups = df.iloc[dup_indices].copy()
    dups["deal_id"] = dups["deal_id"] + "-DUP"
    dups["company_name"] = dups["company_name"].apply(lambda x: dirty_company_name(str(x)) if x else x)
    df = pd.concat([df, dups], ignore_index=True)

    # 2. Scramble some date formats within the same column
    # (already done in messy_date per record — intentional)

    # 3. Inject a handful of completely blank rows (1–2%)
    # Convert bool columns to object first so None can be assigned
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(object)
    for _ in range(int(len(df) * 0.015)):
        blank_idx = random.randint(0, len(df) - 1)
        for col in random.sample(df.columns.tolist(), k=random.randint(8, 15)):
            df.at[blank_idx, col] = None

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────
# DATASET 2: ACTIVE PIPELINE DEALS
# ─────────────────────────────────────────────

def generate_active_pipeline(n=40):
    records = []
    deal_counter = 2000

    # Mix of healthy deals and deals showing stall signals
    stall_mix = (
        [None] * 15 +                          # healthy
        ["Champion Collapse"] * 6 +
        ["Competitor Displacement"] * 6 +
        ["Ghost Stall"] * 7 +
        ["Exec Vacuum"] * 6
    )
    random.shuffle(stall_mix)
    stall_mix = stall_mix[:n]

    for i, stall in enumerate(stall_mix):
        deal_id = f"PIPE-{deal_counter + i}"
        rec = build_deal(deal_id, closed=False, force_stall=stall)
        # Remove outcome fields that wouldn't exist on active deals
        rec["outcome"] = None
        rec["stated_loss_reason"] = None
        rec["sales_cycle_days"] = None
        rec["close_date"] = None
        records.append(rec)

    df = pd.DataFrame(records)

    # Light data quality issues on pipeline too
    dup_indices = random.sample(range(len(df)), k=max(1, int(len(df) * 0.025)))
    dups = df.iloc[dup_indices].copy()
    dups["deal_id"] = dups["deal_id"] + "-DUP"
    df = pd.concat([df, dups], ignore_index=True)

    return df.sample(frac=1, random_state=7).reset_index(drop=True)


# ─────────────────────────────────────────────
# METADATA FILE
# ─────────────────────────────────────────────

def generate_metadata():
    meta = {
        "generated_at": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "Deal Trajectory Engine — Mock Data Generator v1.0",
        "note": "All data is synthetic. Deliberately imperfect: includes duplicates, missing fields, inconsistent date formats, and intentional contradictions between stated_loss_reason and injected_stall_signature.",
        "datasets": {
            "historical_deals.csv": {
                "rows_approx": 150,
                "purpose": "Training set for Win/Loss Synthesizer. Contains closed won and lost deals with behavioral signals.",
                "key_fields": ["outcome", "stated_loss_reason", "injected_stall_signature"],
                "known_issues": [
                    "~3% duplicate deal_ids (suffixed -DUP)",
                    "~8% of non-critical fields randomly nulled",
                    "~45% of lost deals have stated_loss_reason that contradicts injected_stall_signature (intentional ground-truth test)",
                    "Date fields use mixed formats (YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YYYY)",
                    "company_name has inconsistent casing and suffix formatting"
                ]
            },
            "active_pipeline.csv": {
                "rows_approx": 40,
                "purpose": "Live deals for Stall Predictor scoring. No outcome column.",
                "stall_signature_mix": {
                    "None (healthy)": "~38%",
                    "Champion Collapse": "~15%",
                    "Competitor Displacement": "~15%",
                    "Ghost Stall": "~17%",
                    "Exec Vacuum": "~15%"
                },
                "known_issues": [
                    "~2.5% duplicate rows",
                    "Missing fields match same null rate as historical set"
                ]
            }
        },
        "stall_signatures": STALL_SIGNATURES,
        "field_dictionary": {
            "deal_id": "Unique deal identifier",
            "injected_stall_signature": "Ground truth label — which stall pattern was injected into behavioral fields. NULL = healthy or won deal. Use for validation only.",
            "stated_loss_reason": "What the rep logged as the loss reason. Intentionally contradicts injected_stall_signature ~45% of the time.",
            "email_response_latency_delta_days": "Change in email response time from start to end of deal. Positive = slowing down.",
            "champion_engagement_score_delta": "Change in champion engagement score. Negative = disengaging.",
            "stakeholders_at_midpoint": "Unique contacts engaged at cycle midpoint. Low value = over-reliance on single champion.",
            "days_since_multithread_touch": "Days since last outreach to a non-champion contact.",
            "economic_buyer_meetings": "Total meetings with economic buyer / financial approver.",
        }
    }
    return meta


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating historical deals...")
    hist = generate_historical_deals(n=150)
    hist.to_csv("/home/claude/historical_deals.csv", index=False)
    print(f"  → {len(hist)} rows written to historical_deals.csv")
    print(f"  → Outcome split: {hist['outcome'].value_counts().to_dict()}")
    print(f"  → Stall signatures: {hist['injected_stall_signature'].value_counts(dropna=False).to_dict()}")

    print("\nGenerating active pipeline...")
    pipe = generate_active_pipeline(n=40)
    pipe.to_csv("/home/claude/active_pipeline.csv", index=False)
    print(f"  → {len(pipe)} rows written to active_pipeline.csv")
    print(f"  → Stall signature mix: {pipe['injected_stall_signature'].value_counts(dropna=False).to_dict()}")

    print("\nWriting metadata...")
    with open("/home/claude/data_metadata.json", "w") as f:
        json.dump(generate_metadata(), f, indent=2)
    print("  → data_metadata.json written")

    print("\nColumn inventory:")
    print(f"  → {len(hist.columns)} fields per record")
    for col in hist.columns:
        null_pct = round(hist[col].isna().mean() * 100, 1)
        print(f"     {col:<50} {null_pct}% null")
