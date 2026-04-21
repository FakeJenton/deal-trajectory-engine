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
payload.intervention_queue.forEach(d => { d.has_intervention = true; });
payload.non_flagged_deals = nonFlagged;
payload.generated_at = payload.generated_at; // preserved

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
