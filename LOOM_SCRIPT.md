# Deal Trajectory Engine — Loom Walkthrough Script
# Target: ≤ 5 minutes | ~700 words spoken | 4 screen segments

---

## PRE-RECORDING SETUP

Have these open and ready before hitting record:

1. Terminal in the project directory — ready to run `python synthesizer.py`
2. historical_deals.csv open in a spreadsheet viewer (scroll to show messy data)
3. contradiction_report.csv open in a second tab
4. [deal-trajectory-engine.vercel.app](https://deal-trajectory-engine.vercel.app/) open in a browser (live deployment; also available locally as `dashboard.html`)
5. manager_brief.md open in a markdown viewer
6. README.md open as a fallback reference

Keep a timer visible. Aim to hit each section transition on cue.

---

## SECTION 1 — The problem and the framing [0:00 – 0:45]

**Screen:** README.md — scroll slowly past the architecture diagram

**Say:**

"The problem I'm solving is simple to describe and annoying to fix: by the time a rep knows a deal is dying, the behavioral window to do anything about it has already closed. CRMs tell you a deal went dark — they don't tell you it started going dark six weeks ago, and they definitely don't tell you why or what to do about it.

The obvious fix is better dashboards. I didn't build a dashboard. I built something that works backward from historical outcomes — specifically, what did deals that stalled actually look like in the behavioral data before they stalled — and uses those patterns to score your active pipeline and tell a rep exactly what to do today.

The framing that unlocked this: it's not a data quality problem. It's a behavioral drift detection problem. The signal is already in your CRM. It just lives in the interaction between fields, not in any single one."

**Transition cue:** "Let me show you the data layer first."

---

## SECTION 2 — The data and why the messiness matters [0:45 – 1:30]

**Screen:** Switch to historical_deals.csv — scroll right to show field breadth, then scroll down to show mixed date formats and the injected_stall_signature column. Then switch to contradiction_report.csv, filter to is_contradiction = True, show a specific row.

**Say:**

"Phase 1 generates 1,030 historical deals across 97 fields. I made it deliberately messy — mixed date formats in the same column, inconsistent company name casing, ~8% random nulls on behavioral fields. If your data is too clean, you're not testing your own system.

The most important design decision in the data layer is this column — `injected_stall_signature`. This is the ground truth label I buried in the data. And this column — `stated_loss_reason` — is what the rep logged. I built in a ~45% contradiction rate between them on purpose.

Here's a specific example." [point to a row] "Behavioral data screams Champion Collapse — email response latency up 6 days, single stakeholder engaged, no exec access. Rep logged: Price / Budget. The whole point of Phase 2 is catching exactly that."

**Transition cue:** "Let me run the synthesizer."

---

## SECTION 3 — The synthesizer running live [1:30 – 3:00]

**Screen:** Terminal — run `python synthesizer.py` live, let the output stream. Watch for the key lines and call them out as they appear.

**Say:**

[as output streams] "Phase 2 is the engine's brain. It's running Welch's t-tests and Cohen's d across 29 behavioral signals to find what actually separates wins from losses — not what reps say separated them.

[when GBM line appears] Here's the model performance — AUC of 0.826. That means the behavioral signals alone predict win or loss with strong accuracy, before any LLM is involved. That was important to me — I didn't want the intelligence to be 'LLM please analyze this deal.' I wanted explicit, testable logic with statistics behind it.

[when contradiction line appears] 198 contradictions flagged out of 677 lost deals with stated reasons — that's 29%. Nearly a third of your pipeline intelligence is lying to you. The most common one: 'Price / Budget' logged when the behavioral pattern is Exec Vacuum or Champion Collapse.

[when signatures line appears] Four stall signatures derived from the data. Thresholds aren't hardcoded — they're set at the midpoint between median stalled deal and median healthy won deal for each signal. That was a deliberate choice over percentile cutoffs because it's more interpretable to a RevOps person who needs to audit or adjust them."

**Transition cue:** "Now let me show what that does to the active pipeline."

---

## SECTION 4 — The dashboard and a specific deal card [3:00 – 4:30]

**Screen:** Switch to [deal-trajectory-engine.vercel.app](https://deal-trajectory-engine.vercel.app/) (or `dashboard.html` locally). Show the top-level stats briefly, then filter to "Champion Collapse" or click directly on Berry, Garcia and Smith Inc (PIPE-2037) — a critical deal with a contradiction flag. Expand the row and walk through the card.

**Say:**

"Phase 3 scores 150 active deals against those signatures. 94 flagged, $15.6M ACV at risk. The priority sort isn't just stall score — it's stall severity times log of ACV times urgency multiplier. A deal in Negotiation showing Ghost Stall signals outranks the same pattern in Discovery because the intervention window is narrower.

[click the deal] Here's one I want to show you specifically because it has something interesting. Berry, Garcia and Smith. Enterprise deal in Discovery. Champion Collapse — 5 of 6 signals breached. Email response latency up 5.8 days, champion engagement dropped 23 points, single stakeholder at midpoint.

But look at the contradiction flag." [point to it] "Rep notes suggest competitor displacement. Behavioral data says champion collapse. Those are completely different interventions. One is 'go deeper on technical differentiation.' The other is 'stop talking to your champion and find someone else.' If the manager goes into that coaching conversation without seeing this flag, they're coaching the wrong problem.

[scroll to first move] This is the output the rep sees. Specific, drafted, named. Not 'consider multi-threading' — 'map the org and reach out directly with this opening line.'"

**Transition cue:** "Last piece — the manager brief."

---

## SECTION 5 — Close and punts [4:30 – 5:00]

**Screen:** Switch to manager_brief.md — show the top section briefly, then switch back to README.md — Known Limitations section.

**Say:**

"The daily brief bundles the top 10 actions with the contradiction count and rep-level flag distribution. The idea is this hits a Slack channel or email every morning before pipeline review.

The thing I punted on that I most want to build: the feedback loop. Right now the stall signatures are trained once. In production, every intervention you take gets logged, and outcomes feed back into the library — so signatures tighten over time and the thresholds self-adjust. That's the thing that turns this from a pattern-matching tool into one that actually gets smarter. I described the architecture in the README and called it the right first 10-hour next step.

That's the engine."

---

## PACING NOTES

- Section 1: brisk — you're establishing the frame, not demoing yet
- Section 2: slow down on the contradiction example — this is the most important conceptual point
- Section 3: let the terminal breathe — don't talk over the key output lines, pause and point
- Section 4: the deal card is the money shot — take your time here
- Section 5: land the feedback loop punt confidently — this shows systems thinking, not incompleteness

## IF YOU GO LONG

Cut Section 5 to 20 seconds: "One deliberate punt — the feedback loop. Stall signatures train once right now. Production version retrains on intervention outcomes. That's the 10-hour next step. README has the architecture."

## IF YOU GO SHORT

Add to Section 3: "I almost built this as a rule engine with hardcoded thresholds — if days_since_last_activity > 14 then flag. I moved away from that because it's not defensible. If someone asks 'why 14 and not 12?', you don't have an answer. Deriving from data means the threshold reflects actual outcome distributions, not a guess."
