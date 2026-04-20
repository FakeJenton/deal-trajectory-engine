// Build a self-contained dashboard.html by embedding dashboard_payload.json
const fs = require('fs');
const path = require('path');

const payload = fs.readFileSync(path.join(__dirname, 'dashboard_payload.json'), 'utf8');
const template = fs.readFileSync(path.join(__dirname, 'dashboard_template.html'), 'utf8');

const output = template.replace('/*__DATA__*/', 'const DATA = ' + payload + ';');
fs.writeFileSync(path.join(__dirname, 'dashboard.html'), output);
fs.writeFileSync(path.join(__dirname, 'index.html'), output); // For Vercel static hosting

const kb = (fs.statSync(path.join(__dirname, 'dashboard.html')).size / 1024).toFixed(1);
console.log(`Built dashboard.html + index.html (${kb} KB each) with ${JSON.parse(payload).intervention_queue.length} deals embedded`);
