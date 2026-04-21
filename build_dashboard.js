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
