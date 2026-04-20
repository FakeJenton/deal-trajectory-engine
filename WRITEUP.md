# Deal Trajectory Engine — Submission Writeup

## Why this problem

Most pipeline tools tell you a deal is dying after it's already dead. Reps close the CRM tab, log "Price / Budget," and move on. The actual cause — champion collapse, exec vacuum, competitive displacement, a stall that started six weeks ago — gets lost. What you're left with is a loss reason database that's 30% fiction and a rep population that doesn't know why they're losing.

The obvious solution is better dashboards. The obvious solution is wrong. Reps don't need more information about their pipeline — they need to know which deal to touch today and exactly what to do when they touch it. A dashboard with 18 filters and a "days since last activity" column doesn't tell anyone to call the VP of Engineering at Hubbard Group before noon.

The non-obvious framing: this is a behavioral drift detection problem, not a data quality problem. The signal that a deal is failing exists in the CRM weeks before the rep notices — it just lives in the interaction between fields, not in any individual field. Email response latency going up 5 days matters differently depending on whether stakeholder breadth is shrinking at the same time. That's what a single-signal alert can't catch and what a pattern-matching engine can.

## What I built

A four-phase pipeline that runs from raw CRM data to a prioritized, rep-personalized intervention queue.

**Phase 1** generates 1,030 historical deals and 150 active pipeline deals across 97 fields — deliberately messy, with mixed date formats, duplicate rows, missing values, and intentional contradictions between stated loss reasons and behavioral signals.

**Phase 2 (Win/Loss Synthesizer)** loads historical deals, runs Welch's t-tests and Cohen's d on 29 behavioral signals, trains a gradient boosting model (AUC 0.826), and derives data-grounded thresholds for four stall signatures: Champion Collapse, Competitor Displacement, Ghost Stall, and Exec Vacuum. It then cross-references each lost deal's behavioral fingerprint against its stated loss reason and flags the 29.2% of deals where the rep's diagnosis doesn't match the evidence. The most common contradiction: "Price / Budget" logged when the deal shows unmistakable Champion Collapse or Exec Vacuum patterns.

**Phase 3 (Stall Predictor)** scores each active deal against the signature library, applies urgency multipliers based on position in the sales cycle, enforces stage gates (Exec Vacuum doesn't fire on Prospecting deals — that's normal, not a stall), and generates a priority-sorted intervention queue. Classification accuracy against ground truth labels: 77.9%.

**Phase 4 (Intervention Card Generator)** takes the queue and produces rep-specific first moves via the Anthropic API, a daily manager brief, and an interactive dashboard. Each card includes the behavioral evidence with exact threshold values, historical context from comparable deals, a contradiction flag if the rep's notes conflict with the signals, and a drafted first-touch message.

## What I deliberately didn't build

**A feedback loop.** In production, every intervention that gets taken gets logged, outcomes flow back into the pattern library, and stall signatures sharpen over time. I described the architecture and punted the build — it's the right 10-hour next step but would have consumed the entire budget here.

**A UI for threshold tuning.** The thresholds are data-derived but a RevOps manager needs to be able to adjust them without touching Python. That's a configuration layer I skipped.

**Multi-signal alert suppression.** Right now a deal can trigger two stall signatures simultaneously (e.g., Ghost Stall and Exec Vacuum at the same time). The system surfaces the dominant one, but production would need a deduplication and sequencing layer.

**CRM connectors.** Everything runs on CSV. Real deployment would need Salesforce/HubSpot APIs and a scheduler.

## Where it breaks at 10x the data

The GBM retrains on the full historical set each run — at 10,000+ deals that becomes slow enough to matter. Fix: train offline on a schedule, serialize the model, load at score time. The contradiction detection runs a full cross-score on every lost deal at query time — same issue at scale, same fix.

The stall signature thresholds are derived from global medians. At 10x data with proper segmentation, you'd want per-segment, per-industry thresholds — Champion Collapse in Enterprise SaaS looks different than in SMB. The current implementation does segment breakdowns in analysis but applies global thresholds in scoring.

The intervention card generator makes one LLM call per flagged deal. At 1,000 active deals with 60% flagged, that's 600 API calls per run. That needs batching, caching on deal fingerprint hash, and only regenerating when signals change materially.

## What I'd do next with 10 hours

Build the feedback loop. Specifically: log every intervention card that gets generated with a timestamp, track whether the deal subsequently progressed or died, and use that signal to tune the stage-gate thresholds and urgency multipliers with real outcome data rather than the approximate values I'm using now. That's the thing that turns this from a pattern-matching tool into one that actually gets smarter.

Second priority: the rep voice layer. Right now the first-move drafts are pattern-typed by stall signature. With 10 hours I'd pull a sample of each rep's actual sent emails, fine-tune the prompt per rep, and generate drafts that sound like the person sending them — which is the difference between a tool reps use and one they ignore.
