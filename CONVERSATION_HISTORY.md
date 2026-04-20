# Deal Trajectory Engine — Full Conversation History
# Jacob Fenton & Claude | April 20, 2026

---

## Message 1 — Jacob

You are a GTM engineer and are applying to work at a company called Tiger Data. I will attach the take home test they have sent to be completed.

---

## Message 2 — Claude

Sounds good! Go ahead and attach the take home test and I'll dig in.

---

## Message 3 — Jacob

[Attached: Rev_Ops_GTM_Engineer___Take-Home.docx]

I would like for you to first begin with analyzing the options in the 'Option 1' section and determine which would be the greatest opportunity to show your knowledge. I would then like you to review the 'Option 2 - Propose your problem' and suggest a top three projects that would provide the best knowledge and submission. Review the entire document before answering.

---

## Message 4 — Claude

[Read the take-home document, then provided full analysis]

**Option 1 — Best Opportunity Assessment**

The eight starting points vary significantly in how much "judgment" they actually encode vs. how much they're just LLM wrappers. The prompt is explicit: they want scoring, thresholds, tradeoffs, and explicit handling of the ambiguous case — not "LLM please summarize this thing." Here's how I'd rank the top candidates:

**#1 Pick: Outbound Reply Handler** — strongest because it forces multiple distinct judgment layers in compact scope: classification, routing, drafting in the rep's voice, and logging outcomes back. Most demonstrable in a Loom. Maps directly to AI agent work.

**#2: Deal Desk (approve / escalate / reject)** — cleanest scope. Forces threshold definition and escalate cases. Tight, shippable in 2 hours.

**#3: Renewal Triage Tool** — strong given multi-signal scoring background. Weakness: somewhat well-trodden concept.

**Skip:** Morning brief generator and handoff bot risk sliding into "LLM summarize this" territory.

**Option 2 — Top 3 Proposed Problems**

**#1: Technical Champion Signal Engine** — synthesizes GitHub activity, community contributions, job title trajectory, LinkedIn engagement to surface the emergent technical decision-maker in a target account. Novel framing: influence mapping under uncertainty, not enrichment.

**#2: Deal Stall Predictor with Intervention Playbook** — looks at activity pattern deviation vs. comparable won deals at the same stage. Predicts behavioral drift. Output is an intervention card, not a dashboard. Novel framing: predicting behavioral drift, not scoring static attributes.

**#3: Win/Loss Pattern Synthesizer** — synthesizes call recordings, CRM notes, exit survey free text into structured patterns. Key output: which variables actually correlate with win rate, and where stated loss reasons are proxies for the real reason. Novel framing: lie detector for your own pipeline data.

Bottom line recommendation: Option 2 #1 (Technical Champion Signal Engine) for highest ceiling. Safest sharp Option 1 execution: outbound reply handler.

---

## Message 5 — Jacob

Is there a way to combine #2 and #3 from option 2 into one project. I like aspects of both and think they would merge together.

---

## Message 6 — Claude

Yes, and the combination is stronger than either one individually. They're two halves of the same feedback loop:

- Win/Loss Synthesizer answers: "What behavioral patterns separated deals we won from deals we lost?"
- Stall Predictor answers: "Which active deals are drifting toward the losing pattern?"

Combined: a **Deal Trajectory Engine** — it learns the behavioral fingerprint of lost deals from historical data, monitors active pipeline for drift toward those patterns, surfaces the specific warning signs, and generates an intervention recommendation grounded in what actually worked in comparable won deals.

The output card reads something like: "This deal is showing 3 of the 4 signals that preceded your last 6 losses against [Competitor X] in the mid-market segment. In the 4 comparable deals you won, the turning point was executive re-engagement within a week. Recommended action: [specific draft]."

The scoping approach: use a small batch of mock historical deals to derive 3–4 explicit "stall signatures," hardcode those as the pattern library, spend the majority of time on active monitoring and intervention logic. Describe how the pattern library would be generated dynamically at scale in the writeup.

---

## Message 7 — Jacob

So can you now outline the steps of the total project using this Deal Trajectory Engine project idea. My thought is that win/loss synthesizer is a way to determine the best process optimization and then allows us to implement those insights towards the deals that may be showing stall predictors.

---

## Message 8 — Claude

[Provided full project outline across 5 phases]

**The Core Loop:** Historical deals teach the system what winning and losing look like behaviorally. That knowledge gets applied in real time to active deals, so reps get interventions grounded in actual outcomes.

**Phase 1: Data Layer** — Generate historical closed deals and active pipeline deals, deliberately messy.

**Phase 2: Win/Loss Synthesizer** — Pattern extraction (signal divergence points between won/lost), contradiction detection (behavioral pattern vs. stated loss reason), intervention library (what moves preceded wins in comparable recovered deals).

**Phase 3: Stall Predictor** — Signal scoring against signature library, trajectory tagging (healthy/watch/at-risk/critical), contradiction check on rep notes.

**Phase 4: Intervention Card Generator** — Per-deal output card with signal evidence, historical context, contradiction flag, specific recommended action.

**Phase 5: Feedback Loop (describe, don't build)** — Every intervention logged, outcomes feed back into pattern library. The 10-hour next step.

---

## Message 9 — Jacob

Can you complete phase 1 by making the mock generation of data sets. Include any possible field that you think would be valuable to capture in a CRM database.

---

## Message 10 — Claude

[Built and ran generate_mock_data.py, produced initial datasets]

Generated:
- historical_deals.csv: 154 rows, 97 fields
- active_pipeline.csv: 41 rows, 97 fields
- data_metadata.json

97 fields spanning: identifiers, deal metadata, financials, timeline, contacts & stakeholders, email & communication signals, meeting & call signals, multi-thread/engagement breadth, competitive & objection signals, validation & process signals, product/trial signals, intent & 3rd-party signals, free text/qualitative fields.

Key design decisions: ~3% duplicate rows, mixed date formats, inconsistent company name casing, random nulls at realistic rates (8% on most fields), free-text rep notes with intentional typos. The contradiction layer: ~45% of lost deals have a stated_loss_reason that doesn't match the injected_stall_signature.

---

## Message 11 — Jacob

I would like the code to generate 100 rows of active deals in the active pipeline data and 500 rows of historical deals in the historical deals data. If you do not believe that these will be enough to provide the project AI and build with meaningful insights and recognitions that are not easily identified by an employee feel free to increase.

---

## Message 12 — Claude

[Recommended and built scaled-up version: 1,000 historical + 150 active pipeline]

Reasoning: With 500 historical rows and ~55% lost rate, slicing to "Mid-Market Champion Collapse deals lost to Snowflake" yields only 8–12 rows — too thin for non-obvious pattern detection. At 1,000 rows the same cross-cut yields ~46 rows per cell. Recommended 1,000 historical and 150 active.

Validated cross-cuts:
- Champion Collapse densest in Mid-Market (51 losses vs ~37-42 in other segments)
- Competitor Displacement evenly distributed — positioning problem, not targeting problem
- Ghost Stall almost perfectly split across segments — rep behavior pattern, not account quality
- Exec Vacuum skews disproportionately toward SMB (61 losses) — counterintuitive

Generated: generate_mock_data_v2.py, historical_deals.csv (1,030 rows), active_pipeline.csv (153 rows)

---

## Message 13 — Jacob

Perfect, now lets begin on phase two.

---

## Message 14 — Claude

[Built synthesizer.py — full 8-step Win/Loss Synthesizer pipeline]

Steps:
1. Load & clean — normalize booleans, coerce numerics, drop duplicates, fill nulls
2. Pattern extraction — Welch's t-tests + Cohen's d on 29 behavioral signals
3. Feature importance — GBM with permutation importance (AUC 0.826 ± 0.043)
4. Stall signature library — data-derived thresholds (midpoint between stall median and healthy median)
5. Contradiction detection — 198 of 677 lost deals (29.2%) flagged
6. Intervention library — recovery rates from behavioral analysis per stall type
7. Non-obvious insight extraction — 5 programmatic findings
8. LLM synthesis report — narrative via Anthropic API

Key results:
- Top 5 predictive signals: days_since_last_activity, email_response_latency_delta_days, email_response_rate, pricing_objection_count, days_since_multithread_touch
- Most common contradiction: "Price / Budget" logged when behavioral evidence points to Exec Vacuum (69 deals) or Champion Collapse (50 deals)
- Recovery rates: Champion Collapse 30%, Ghost Stall 30%, Exec Vacuum 23%, Competitor Displacement 23%

Outputs: win_loss_patterns.json, feature_importance.json, stall_signature_library.json, contradiction_report.csv, intervention_library.json, non_obvious_insights.json, synthesis_report.md

---

## Message 15 — Jacob

Continue.

---

## Message 16 — Claude

[Built stall_predictor.py — Phase 3 Stall Predictor]

Architecture:
- Loads active_pipeline.csv and Phase 2 outputs
- Scores each deal against all 4 stall signatures
- Applies urgency multipliers based on deal cycle position (0.8× early, 1.0× mid, 1.3× late, 1.5× overdue)
- Enforces stage gates (Exec Vacuum doesn't fire on Prospecting/Discovery — normal, not stall)
- Minimum signal gate: must breach ≥2 signals to be tagged at-risk or critical
- Priority score = stall_severity × log(ACV) × urgency_multiplier

Went through 3 calibration iterations:
- Initial run: 84% flagged (too broad — Exec Vacuum's 2 binary signals hitting everything)
- After threshold raise: still 82% flagged
- After stage gating: 62.7% flagged, matching the 63% of active deals with injected stall signatures

Final results:
- 150 deals scored, 94 flagged, $15.6M ACV at risk
- Stall type distribution: Exec Vacuum 27, Competitor Displacement 24, Ghost Stall 22, Champion Collapse 21
- 77.9% stall type classification accuracy against ground truth labels
- 37 contradiction flags (rep notes ≠ behavioral signals)

Outputs: scored_pipeline.csv, flagged_deals.csv, intervention_queue.json, predictor_summary.json

---

## Message 17 — Jacob

Sounds good.

---

## Message 18 — Claude

[Built intervention_card_generator.py — Phase 4 Card Generator + interactive dashboard]

Python script generates:
- Per-deal personalized first-move drafts via Anthropic API (pattern-matched template fallback)
- 20 individual markdown cards for top priority deals (cards/ folder)
- Daily manager brief (manager_brief.md)
- Full dashboard payload JSON (dashboard_payload.json)

Interactive React dashboard built inline showing:
- Pipeline health stats (Total ACV $27.7M, ACV at risk $15.6M, 94/150 flagged, 88 critical, 37 contradictions)
- Filter buttons by stall type
- Expandable deal rows with full intervention cards
- Signal evidence with exact threshold values
- Historical context with recovery rates
- Contradiction flags where rep notes conflict with behavioral data
- Drafted first-move text per rep

Notable deal highlighted: Berry, Garcia and Smith — Champion Collapse with contradiction flag. Rep notes suggest competitor displacement; behavioral data says champion collapse. Completely different interventions.

Outputs: intervention_card_generator.py, manager_brief.md, dashboard_payload.json, cards/ (20 files)

---

## Message 19 — Jacob

Lets go into the write up and README.

---

## Message 20 — Claude

[Built WRITEUP.md and README.md]

**WRITEUP.md (≤1 page submission document):**

Why this problem: standard pipeline tools tell you a deal is dying after it's already dead. The obvious fix (dashboards) is wrong — reps need to know which deal to touch today and exactly what to do. This is a behavioral drift detection problem, not a data quality problem.

What was built: four-phase pipeline from raw CRM data to prioritized intervention queue.

What was deliberately not built: feedback loop (right 10-hour next step), threshold tuning UI, multi-signature deduplication, CRM connectors.

Where it breaks at 10x: GBM retrains on full historical set each run — slow at 10,000+ deals, fix is offline training with serialized model. Contradiction detection runs full cross-score at query time — same fix. Stall signature thresholds are global medians — production needs per-segment thresholds. 600 LLM API calls per run at scale — needs batching and fingerprint-hash caching.

10-hour next step: the feedback loop. Log every intervention card, track outcomes, use signal to tune thresholds. Turns a pattern-matching tool into one that gets smarter. Second priority: rep voice layer — fine-tune prompts per rep's actual sent emails.

**README.md:** Full technical runbook. Setup, run instructions, data field dictionary, stall signature descriptions, scoring logic, model performance, known limitations, complete file index.

---

## Message 21 — Jacob

Sounds good.

---

## Message 22 — Claude

[Built LOOM_SCRIPT.md — 5-section Loom walkthrough script]

Structure:
- Section 1 (0:00–0:45): Problem framing — "I didn't build a dashboard. I built something that works backward from historical outcomes."
- Section 2 (0:45–1:30): Data layer and contradiction example — show historical_deals.csv field breadth + mixed dates + contradiction_report.csv with a specific row
- Section 3 (1:30–3:00): Run synthesizer.py live — call out AUC 0.826, 198 contradictions flagged, data-derived thresholds vs. hardcoded rule engine
- Section 4 (3:00–4:30): Dashboard + Berry, Garcia and Smith deal card — champion collapse with contradiction flag, the money moment
- Section 5 (4:30–5:00): Manager brief + feedback loop punt landed as confidence not apology

Includes pacing notes, "if you go long" cut, and "if you go short" extension with the rule engine vs. data-derived threshold story.

---

## Message 23 — Jacob

Can I move this entire conversation into Claude Code so I can continue to tweak?

---

## Message 24 — Claude

Yes — explained the transition:

What transfers cleanly: all the files. The project is the files.

What doesn't transfer: conversation history itself. However README.md was written to capture the architecture decisions and design rationale — it's a condensed version of the conversation. Opening Claude Code with the files present and README loaded gives it enough context.

Practical steps:
1. Download all files from outputs folder
2. Create local project directory: deal-trajectory-engine/
3. Drop all files in, keep cards/ subfolder intact
4. Open terminal in directory, run: claude
5. Start session with: "I'm continuing work on this project. Read the README.md first to understand the full context, then I'll tell you what I want to tweak."

Note: Set ANTHROPIC_API_KEY in environment. Interactive dashboard artifact lives in this Claude.ai conversation — to replicate in Claude Code, ask it to generate an HTML file version that opens in a browser.

---

## Message 25 — Jacob

Can you provide me a folder of all the files that have been generated in this chat and then provide a document with the history of the actual dialogue between you and I.

---

## Message 26 — Claude

[This response — building the zip archive and this conversation history document]

---

## APPENDIX — Complete File Inventory

| File | Phase | Description |
|------|-------|-------------|
| generate_mock_data.py | 1 | Original data generator (v1, 150 historical / 40 active) |
| generate_mock_data_v2.py | 1 | Scaled data generator (1,000 historical / 150 active) |
| historical_deals.csv | 1 | 1,030 closed deals, 97 fields, deliberately messy |
| active_pipeline.csv | 1 | 153 active pipeline deals, 97 fields |
| data_metadata.json | 1 | Field dictionary, stall signature definitions, known data issues |
| synthesizer.py | 2 | Win/Loss Synthesizer — 8-step pipeline |
| win_loss_patterns.json | 2 | Statistical comparison of 29 signals (won vs. lost) |
| feature_importance.json | 2 | GBM permutation importance, top 20 signals |
| stall_signature_library.json | 2 | Data-derived thresholds for 4 stall signatures |
| contradiction_report.csv | 2 | 677 lost deals, 198 contradictions flagged |
| intervention_library.json | 2 | Recovery playbook per stall type |
| non_obvious_insights.json | 2 | 5 programmatic non-obvious findings |
| synthesis_report.md | 2 | LLM-generated narrative summary (fallback template) |
| stall_predictor.py | 3 | Stall Predictor — scores active pipeline |
| scored_pipeline.csv | 3 | All 150 active deals with scores and trajectory tags |
| flagged_deals.csv | 3 | 94 at-risk + critical deals, sorted by priority |
| intervention_queue.json | 3 | 94 priority-sorted intervention cards |
| predictor_summary.json | 3 | Aggregate pipeline stats for reporting |
| intervention_card_generator.py | 4 | Card generator + manager brief builder |
| manager_brief.md | 4 | Daily manager brief, top 10 priority actions |
| dashboard_payload.json | 4 | Full serialized output for interactive dashboard |
| cards/PIPE-XXXX.md (x20) | 4 | Individual intervention cards, top 20 by priority |
| WRITEUP.md | Submission | 1-page submission writeup |
| README.md | Submission | Full technical runbook |
| LOOM_SCRIPT.md | Submission | 5-section Loom walkthrough script with timing |
| CONVERSATION_HISTORY.md | Reference | This document |
