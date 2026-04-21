// Build a self-contained dashboard.html + index.html by:
//   1. Reading scored_pipeline.csv to include ALL 150 deals (not just 94 flagged)
//   2. Extending dashboard_payload.json with non-flagged deals (watch / healthy)
//   3. Embedding the full payload into dashboard_template.html
//
// Run with: node build_dashboard.js
const fs = require('fs');
const path = require('path');

// ------------------------------ CSV parser ------------------------------
function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += c; i++;
    } else {
      if (c === '"') { inQuotes = true; i++; continue; }
      if (c === ',') { row.push(field); field = ''; i++; continue; }
      if (c === '\n') { row.push(field); field = ''; rows.push(row); row = []; i++; continue; }
      if (c === '\r') { i++; continue; }
      field += c; i++;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// ------------------------------ Load inputs ------------------------------
const payload = JSON.parse(fs.readFileSync(path.join(__dirname, 'dashboard_payload.json'), 'utf8'));
const csvRaw = fs.readFileSync(path.join(__dirname, 'scored_pipeline.csv'), 'utf8');
const rows = parseCSV(csvRaw);
const header = rows[0];
const idx = Object.fromEntries(header.map((h, i) => [h, i]));

const flaggedIds = new Set(payload.intervention_queue.map(d => d.deal_id));

// Index CSV rows by deal_id so we can stamp extra fields onto the intervention queue too
const csvById = {};
for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (!row.length || !row[idx.deal_id]) continue;
  csvById[row[idx.deal_id]] = row;
}

// Load historical analysis artifacts
const featureImportance = JSON.parse(fs.readFileSync(path.join(__dirname, 'feature_importance.json'), 'utf8'));
const stallLibrary = JSON.parse(fs.readFileSync(path.join(__dirname, 'stall_signature_library.json'), 'utf8'));
const interventionLib = JSON.parse(fs.readFileSync(path.join(__dirname, 'intervention_library.json'), 'utf8'));
const nonObviousInsights = JSON.parse(fs.readFileSync(path.join(__dirname, 'non_obvious_insights.json'), 'utf8'));
const winLossPatterns = JSON.parse(fs.readFileSync(path.join(__dirname, 'win_loss_patterns.json'), 'utf8'));

// Parse contradiction_report.csv for cross-tab
const contrRaw = fs.readFileSync(path.join(__dirname, 'contradiction_report.csv'), 'utf8');
const contrRows = parseCSV(contrRaw);
const contrHeader = contrRows[0];
const contrIdx = Object.fromEntries(contrHeader.map((h, i) => [h, i]));

// ------------------------------ Extract non-flagged deals ------------------------------
const nonFlagged = [];
for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (!row.length || !row[idx.deal_id]) continue;
  const dealId = row[idx.deal_id];
  if (flaggedIds.has(dealId)) continue;

  const traj = row[idx.trajectory] || 'healthy';
  const acv = parseFloat(row[idx.acv_usd]) || 0;
  const signalsHit = parseInt(row[idx.signals_hit_count]) || 0;
  // Only surface a dominant stall type if the deal actually hit signals.
  // Otherwise scored_pipeline.csv records the nearest stall type which is noise for healthy deals.
  const stallType = signalsHit > 0 ? (row[idx.dominant_stall_type] || null) : null;
  nonFlagged.push({
    deal_id: dealId,
    company_name: row[idx.company_name] || '—',
    rep_name: row[idx.rep_name] || '—',
    segment: row[idx.segment] || '—',
    acv_usd: acv,
    current_stage: row[idx.current_stage] || '—',
    trajectory: traj,
    dominant_stall_type: stallType,
    priority_score: parseFloat(row[idx.priority_score]) || 0,
    has_intervention: false,
    // Healthy/watch deals have no playbook card. Pass enough signal info for the expanded view:
    signals_hit_count: signalsHit,
    signals_hit_labels: row[idx.signals_hit_labels] || '',
    next_step_defined: row[idx.next_step_defined] === 'True' || row[idx.next_step_defined] === 'true',
    forecast_category: row[idx.forecast_category] || '—',
  });
}
nonFlagged.sort((a, b) => b.acv_usd - a.acv_usd);

// Mark the flagged deals too so the frontend can tell them apart
payload.intervention_queue.forEach(d => {
  d.has_intervention = true;
  const row = csvById[d.deal_id];
  if (row) {
    d.days_in_current_stage = parseInt(row[idx.days_in_current_stage], 10) || 0;
    if (d.signals_hit_labels == null) d.signals_hit_labels = row[idx.signals_hit_labels] || '';
    if (d.signals_hit_count == null) d.signals_hit_count = parseInt(row[idx.signals_hit_count], 10) || 0;
  }
});
// Also stamp days_in_current_stage onto non-flagged entries
nonFlagged.forEach(d => {
  const row = csvById[d.deal_id];
  if (row) d.days_in_current_stage = parseInt(row[idx.days_in_current_stage], 10) || 0;
});
payload.non_flagged_deals = nonFlagged;
payload.generated_at = payload.generated_at; // preserved

// ------------------------------ Enrichment: confidence, review status, composite priority, score factors ------------------------------
// Keep this layer simple and explicit — scoring rules live here, rendering stays dumb.

const RECOVERY_BY_STALL = {};
for (const [stallType, lib] of Object.entries(stallLibrary)) {
  const intr = interventionLib[stallType] || {};
  if (intr.recovery_rate != null) RECOVERY_BY_STALL[stallType] = intr.recovery_rate;
}
const AVG_RECOVERY = Object.values(RECOVERY_BY_STALL).length
  ? Object.values(RECOVERY_BY_STALL).reduce((a, b) => a + b, 0) / Object.values(RECOVERY_BY_STALL).length
  : 0.3;

const LATE_STAGES = new Set(['Negotiation', 'Proposal', 'Procurement', 'Legal Review']);
const MID_STAGES  = new Set(['Technical Validation', 'Demo', 'Evaluation']);

// ----- confidence -----
function scoreConfidence(d) {
  let c = 1.0;
  const factors = [];

  // Data completeness
  const missing = [];
  if (!d.forecast_category || d.forecast_category === '—') missing.push('forecast category');
  if (d.next_step_defined === false) missing.push('next step');
  if (!d.current_stage || d.current_stage === '—') missing.push('current stage');
  if (!d.rep_name || d.rep_name === '—') missing.push('deal owner');
  if (missing.length) {
    const drop = Math.min(0.35, 0.12 * missing.length);
    c -= drop;
    factors.push({ label: 'Missing data', detail: missing.join(', '), delta: -drop });
  }

  // Evidence strength (flagged deals carry signal_coverage or signals_hit_count)
  const sigHit = d.signals_hit_count || (d.signal_coverage ? parseInt(d.signal_coverage.split('/')[0], 10) : 0);
  const sigTotal = d.signal_coverage ? parseInt(d.signal_coverage.split('/')[1], 10) || 6 : 6;
  if (d.has_intervention) {
    if (sigHit && sigHit <= 2) {
      c -= 0.15;
      factors.push({ label: 'Thin evidence', detail: `only ${sigHit} of ${sigTotal} signals breached`, delta: -0.15 });
    } else if (sigHit && sigHit >= 5) {
      c += 0.05;
      factors.push({ label: 'Strong evidence', detail: `${sigHit} of ${sigTotal} signals breached`, delta: +0.05 });
    }
  }

  // Conflicting signals (note_contradiction in original payload means behavior disagrees with rep notes)
  if (d.note_contradiction) {
    c -= 0.2;
    factors.push({ label: 'Conflicting signals', detail: 'Rep notes disagree with behavioral data', delta: -0.2 });
  }

  // Similarity to known pattern — if flagged with a recognized stall type, bump confidence
  if (d.has_intervention && d.dominant_stall_type && RECOVERY_BY_STALL[d.dominant_stall_type] != null) {
    c += 0.05;
    factors.push({ label: 'Matches known pattern', detail: `${d.dominant_stall_type} — ${Math.round(RECOVERY_BY_STALL[d.dominant_stall_type] * 100)}% historical save rate`, delta: +0.05 });
  }

  // Clamp
  c = Math.max(0, Math.min(1, c));
  return { confidence_score: +c.toFixed(2), confidence_factors: factors };
}

// ----- review status -----
function reviewStatusFor(d) {
  const c = d.confidence_score;
  const hasContradiction = !!d.note_contradiction;
  const sigHit = d.signals_hit_count || (d.signal_coverage ? parseInt(d.signal_coverage.split('/')[0], 10) : 0);

  if (c < 0.55 || hasContradiction || (d.has_intervention && sigHit && sigHit <= 2)) {
    return {
      review_status: 'escalate',
      review_reason: hasContradiction
        ? 'Behavioral signals conflict with the rep\'s logged notes — a human should reconcile before acting.'
        : sigHit && sigHit <= 2
          ? 'Evidence is thin (only a couple of signals breached). A manager should confirm the read before the rep acts.'
          : 'Data completeness is below the threshold we trust for auto-action. Fill in missing fields or escalate for review.',
    };
  }
  if (c < 0.75) {
    return {
      review_status: 'caution',
      review_reason: 'Recommendation is directionally right but worth a quick sanity check before sending anything to the customer.',
    };
  }
  return {
    review_status: 'auto',
    review_reason: 'High confidence — data is complete, signals are aligned, and the pattern matches known history. Safe to act on.',
  };
}

// ----- composite priority -----
function compositePriority(d) {
  const acv = d.acv_usd || 0;
  const logAcv = acv > 0 ? Math.log10(acv) : 0;        // 3..6 typically
  const risk = d.stall_score != null ? d.stall_score : (d.has_intervention ? 0.5 : 0.1);
  const urgency = LATE_STAGES.has(d.current_stage) ? 1.3
                 : MID_STAGES.has(d.current_stage) ? 1.0
                 : 0.8;
  const recovery = d.dominant_stall_type && RECOVERY_BY_STALL[d.dominant_stall_type] != null
    ? RECOVERY_BY_STALL[d.dominant_stall_type]
    : AVG_RECOVERY;
  // Recoverable value — rewards deals where action is likely to matter
  const recoverable = acv * recovery;
  const logRecoverable = recoverable > 0 ? Math.log10(recoverable) : 0;

  // Composite: risk × urgency × logAcv — but tempered by confidence so low-confidence deals don't top the list,
  // plus a recoverable-value boost so big-dollar recoverable deals outrank equivalent non-recoverable ones.
  const c = d.confidence_score;
  const base = risk * urgency * logAcv;
  const boost = logRecoverable * 0.25;
  const composite = (base + boost) * (0.5 + 0.5 * c);      // conf acts as multiplier 0.5..1.0
  return +composite.toFixed(3);
}

// ----- score factors (top drivers for "why this score") -----
function buildScoreFactors(d) {
  const factors = [];
  const acv = d.acv_usd || 0;

  // Risk
  const risk = d.stall_score != null ? d.stall_score : null;
  if (risk != null) {
    const label = risk >= 0.7 ? 'High risk' : risk >= 0.4 ? 'Moderate risk' : 'Low risk';
    factors.push({ weight: 'primary', text: `${label} — ${Math.round(risk * 100)}% of defining signals breached` });
  } else if (d.has_intervention) {
    factors.push({ weight: 'primary', text: 'Flagged at-risk based on behavioral pattern match' });
  } else {
    factors.push({ weight: 'primary', text: 'On-track — no stall pattern signals breached' });
  }

  // Value
  const valueLabel = acv >= 250000 ? 'High value'
                   : acv >= 100000 ? 'Mid value'
                   : 'Low value';
  factors.push({ weight: 'secondary', text: `${valueLabel} — $${Math.round(acv).toLocaleString()} ACV` });

  // Urgency
  if (LATE_STAGES.has(d.current_stage)) {
    factors.push({ weight: 'secondary', text: `Late stage (${d.current_stage}) — short window to act` });
  } else if (MID_STAGES.has(d.current_stage)) {
    factors.push({ weight: 'tertiary', text: `Mid stage (${d.current_stage}) — normal urgency` });
  } else {
    factors.push({ weight: 'tertiary', text: `Early stage (${d.current_stage}) — longer window` });
  }

  // Recoverability
  if (d.dominant_stall_type && RECOVERY_BY_STALL[d.dominant_stall_type] != null) {
    const rate = RECOVERY_BY_STALL[d.dominant_stall_type];
    const label = rate >= 0.4 ? 'Highly recoverable' : rate >= 0.2 ? 'Moderately recoverable' : 'Hard to recover';
    factors.push({ weight: 'secondary', text: `${label} — ${d.dominant_stall_type} saves ${Math.round(rate * 100)}% historically` });
  }

  // Confidence
  const c = d.confidence_score;
  const cLabel = c >= 0.75 ? 'Confidence strong'
               : c >= 0.55 ? 'Confidence moderate — check before acting'
               : 'Confidence low — human review recommended';
  factors.push({ weight: 'tertiary', text: `${cLabel} (${Math.round(c * 100)}%)` });

  return factors;
}

// ----- apply enrichment to every deal -----
const allDeals = payload.intervention_queue.concat(payload.non_flagged_deals);
for (const d of allDeals) {
  const { confidence_score, confidence_factors } = scoreConfidence(d);
  d.confidence_score = confidence_score;
  d.confidence_factors = confidence_factors;
  const rs = reviewStatusFor(d);
  d.review_status = rs.review_status;
  d.review_reason = rs.review_reason;
  d.composite_priority = compositePriority(d);
  d.score_factors = buildScoreFactors(d);
}

// ----- rerank queues using composite priority with tie-breakers -----
function rankComparator(a, b) {
  // primary: composite priority desc
  if ((b.composite_priority || 0) !== (a.composite_priority || 0)) {
    return (b.composite_priority || 0) - (a.composite_priority || 0);
  }
  // tie-break 1: higher confidence first (so ties don't surface low-confidence noise)
  if ((b.confidence_score || 0) !== (a.confidence_score || 0)) {
    return (b.confidence_score || 0) - (a.confidence_score || 0);
  }
  // tie-break 2: later-stage first (more urgent)
  const stageOrder = { 'Prospecting': 0, 'Discovery': 1, 'Demo': 2, 'Evaluation': 2, 'Technical Validation': 3, 'Proposal': 4, 'Procurement': 5, 'Legal Review': 5, 'Negotiation': 6 };
  const sa = stageOrder[a.current_stage] ?? 1;
  const sb = stageOrder[b.current_stage] ?? 1;
  if (sa !== sb) return sb - sa;
  // tie-break 3: higher ACV first
  return (b.acv_usd || 0) - (a.acv_usd || 0);
}
payload.intervention_queue.sort(rankComparator);
payload.non_flagged_deals.sort(rankComparator);

// Stamp the rank on each deal so the UI can say "ranks #N because …"
const ranked = payload.intervention_queue.concat(payload.non_flagged_deals).sort(rankComparator);
ranked.forEach((d, i) => { d.overall_rank = i + 1; });

// ----- pipeline-level summary for compact summary strip -----
const reviewCounts = { auto: 0, caution: 0, escalate: 0 };
let confSum = 0, confCount = 0, acvEscalate = 0;
for (const d of ranked) {
  reviewCounts[d.review_status] = (reviewCounts[d.review_status] || 0) + 1;
  confSum += d.confidence_score;
  confCount += 1;
  if (d.review_status === 'escalate') acvEscalate += (d.acv_usd || 0);
}
payload.pipeline_summary = {
  review_counts: reviewCounts,
  avg_confidence: confCount ? +(confSum / confCount).toFixed(2) : 0,
  acv_in_escalate: acvEscalate,
  top_deal: ranked[0] ? { deal_id: ranked[0].deal_id, company_name: ranked[0].company_name, acv_usd: ranked[0].acv_usd } : null,
};

// ----- aggregate stats for the CRO-facing header strip -----
let totalAcv = 0, atRiskAcv = 0, criticalCount = 0, contradictionCount = 0;
for (const d of ranked) {
  totalAcv += d.acv_usd || 0;
  if (d.trajectory === 'critical' || d.trajectory === 'at-risk') atRiskAcv += d.acv_usd || 0;
  if (d.trajectory === 'critical') criticalCount += 1;
  if (d.note_contradiction) contradictionCount += 1;
}
payload.aggregate_stats = {
  total_acv_scored: totalAcv,
  acv_at_risk: atRiskAcv,
  critical_count: criticalCount,
  contradiction_count: contradictionCount,
};

// ----- rep breakdown (leader view) — one row per rep with >=1 flagged deal -----
const repMap = {};
for (const d of payload.intervention_queue) {
  const rep = d.rep_name || '—';
  if (!repMap[rep]) {
    repMap[rep] = { rep_name: rep, critical: 0, at_risk: 0, acv_at_risk: 0, stall_counts: {} };
  }
  const row = repMap[rep];
  if (d.trajectory === 'critical') row.critical += 1;
  else if (d.trajectory === 'at-risk') row.at_risk += 1;
  row.acv_at_risk += d.acv_usd || 0;
  const st = d.dominant_stall_type || 'Unknown';
  row.stall_counts[st] = (row.stall_counts[st] || 0) + 1;
}
payload.rep_breakdown = Object.values(repMap).map(r => {
  const topStall = Object.entries(r.stall_counts).sort((a, b) => b[1] - a[1])[0];
  return {
    rep_name: r.rep_name,
    critical: r.critical,
    at_risk: r.at_risk,
    acv_at_risk: r.acv_at_risk,
    top_stall: topStall ? topStall[0] : '—',
    top_stall_count: topStall ? topStall[1] : 0,
  };
}).sort((a, b) => b.acv_at_risk - a.acv_at_risk);

// ------------------------------ Historical analysis payload ------------------------------
// 1. Contradiction cross-tab: stated_loss_reason × behavioral_best_match
const crosstab = {};
const contradictionCounts = { total: 0, contradictions: 0 };
const lossReasonTotals = {};
for (let r = 1; r < contrRows.length; r++) {
  const row = contrRows[r];
  if (!row.length || !row[contrIdx.deal_id]) continue;
  contradictionCounts.total++;
  const stated = row[contrIdx.stated_loss_reason] || 'Unknown';
  const behavioral = row[contrIdx.behavioral_best_match] || 'Unknown';
  const isContradiction = row[contrIdx.is_contradiction] === 'True';
  lossReasonTotals[stated] = (lossReasonTotals[stated] || 0) + 1;
  if (isContradiction) {
    contradictionCounts.contradictions++;
    if (!crosstab[stated]) crosstab[stated] = {};
    crosstab[stated][behavioral] = (crosstab[stated][behavioral] || 0) + 1;
  }
}

// 2. Win/loss signal comparison — keep only significant signals, sort by |Cohen's d|
const signalComparison = Object.entries(winLossPatterns)
  .filter(([, v]) => v.significant)
  .map(([signal, v]) => ({
    signal,
    won_mean: v.won_mean,
    lost_mean: v.lost_mean,
    delta: v.mean_delta,
    cohen_d: v.cohen_d,
    effect_size: v.effect_size,
    loss_direction: v.loss_direction,
    segment_breakdown: v.segment_breakdown || {},
  }))
  .sort((a, b) => Math.abs(b.cohen_d) - Math.abs(a.cohen_d))
  .slice(0, 20);

// 3. Stall library — merge stall_signature_library + intervention_library into one structure
const stallLibraryMerged = {};
for (const [stallType, lib] of Object.entries(stallLibrary)) {
  const interventions = interventionLib[stallType] || {};
  stallLibraryMerged[stallType] = {
    description: lib.description,
    deal_count: lib.deal_count,
    loss_rate: lib.loss_rate,
    avg_acv: lib.avg_acv,
    avg_cycle_days: lib.avg_cycle_days,
    recovery_rate: interventions.recovery_rate,
    recovered_deal_count: interventions.recovered_deal_count,
    lost_deal_count: interventions.lost_deal_count,
    segment_loss_rate: lib.segment_loss_rate || {},
    top_signals: Object.entries(lib.defining_signals || {}).map(([signal, s]) => ({
      signal,
      threshold: s.threshold,
      direction: s.direction,
      stall_median: s.stall_median,
      healthy_median: s.healthy_median,
      cohen_d: s.cohen_d,
    })).sort((a, b) => Math.abs(b.cohen_d) - Math.abs(a.cohen_d)),
    recovery_differentiators: interventions.top_differentiators || [],
    timing_insights: interventions.timing_insights || null,
    interventions: interventions.interventions || [],
  };
}

payload.historical = {
  summary: {
    total_historical_deals: contradictionCounts.total,
    contradictions: contradictionCounts.contradictions,
    contradiction_pct: contradictionCounts.total
      ? +(contradictionCounts.contradictions / contradictionCounts.total * 100).toFixed(1)
      : 0,
    loss_reason_totals: lossReasonTotals,
  },
  feature_importance: featureImportance.slice(0, 12),
  signal_comparison: signalComparison,
  stall_library: stallLibraryMerged,
  contradiction_crosstab: crosstab,
  insights: nonObviousInsights,
};

// ------------------------------ Write updated payload + dashboards ------------------------------
fs.writeFileSync(
  path.join(__dirname, 'dashboard_payload.json'),
  JSON.stringify(payload, null, 2)
);

const template = fs.readFileSync(path.join(__dirname, 'dashboard_template.html'), 'utf8');
const output = template.replace(
  '/*__DATA__*/',
  'const DATA = ' + JSON.stringify(payload) + ';'
);
fs.writeFileSync(path.join(__dirname, 'dashboard.html'), output);
fs.writeFileSync(path.join(__dirname, 'index.html'), output);

const totalDeals = payload.intervention_queue.length + payload.non_flagged_deals.length;
const kb = (fs.statSync(path.join(__dirname, 'dashboard.html')).size / 1024).toFixed(1);
console.log(`Built dashboard.html + index.html (${kb} KB each)`);
console.log(`  ${payload.intervention_queue.length} flagged + ${payload.non_flagged_deals.length} non-flagged = ${totalDeals} total deals`);
