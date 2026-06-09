/* ============================================================
   app.js — Credit Foundation Model Dashboard Logic
   Chart.js charts + FastAPI integration
   ============================================================ */

'use strict';

// -- State --------------------------------------------------─
const state = {
  validationChecks: [],
  modelResults: null,
  stageStatus: {
    ingest:   'idle',
    validate: 'idle',
    train:    'idle',
    tokenize: 'idle',
  },
};

// -- Chart Instances (singleton pattern) --------------------─
const charts = {};

// -- Chart.js Global Defaults --------------------------------
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

const CHART_GRID = {
  color: 'rgba(99,179,237,0.07)',
  drawBorder: false,
};

// -- Utility: format number ----------------------------------─
function fmt(n, dec = 0) {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString('en-US', { maximumFractionDigits: dec });
}

function fmtPct(n, dec = 2) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(dec) + '%';
}

// -- Toast ----------------------------------------------------
function toast(msg, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// -- Global status badge --------------------------------------
function setGlobalStatus(text, cls = 'status-idle') {
  const badge = document.getElementById('global-status');
  const label = document.getElementById('global-status-text');
  badge.className = `status-badge ${cls}`;
  label.textContent = text;
}

// -- Stage status badge --------------------------------------─
function setStageStatus(stage, cls, text) {
  const badge = document.getElementById(`${stage}-status-badge`);
  if (!badge) return;
  badge.className = `status-badge ${cls}`;
  badge.innerHTML = `<span class="dot"></span><span>${text}</span>`;
  state.stageStatus[stage] = cls.replace('status-', '');
}

// -- Append to log box ----------------------------------------
function appendLog(stage, text, cls = '') {
  const box = document.getElementById(`${stage}-log`);
  if (!box) return;
  const line = document.createElement('span');
  if (cls) line.className = cls;
  line.textContent = text;
  box.appendChild(line);
  box.appendChild(document.createTextNode('\n'));
  box.scrollTop = box.scrollHeight;
}

function clearLog(stage) {
  const box = document.getElementById(`${stage}-log`);
  if (box) box.innerHTML = '';
}

// -- Progress bar --------------------------------------------─
function setProgress(stage, pct) {
  const bar = document.getElementById(`${stage}-progress`);
  if (bar) bar.style.width = `${pct}%`;
}

// -- Button loading state ------------------------------------─
function setBtnLoading(id, loading) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.disabled = loading;
  btn.classList.toggle('loading', loading);
}

// -- Tab switching --------------------------------------------
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => {
    el.classList.remove('active');
    el.setAttribute('aria-selected', 'false');
  });
  document.getElementById(`tab-${name}`).classList.add('active');
  const btn = document.getElementById(`tab-${name}-btn`);
  if (btn) { btn.classList.add('active'); btn.setAttribute('aria-selected', 'true'); }
}

// -- Filter validation checks --------------------------------─
function filterChecks() {
  const failOnly = document.getElementById('filter-failed-only').checked;
  renderChecksList(failOnly);
}

// -- Fetch DB status ------------------------------------------
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const data = await r.json();
    updateStatsGrid(data);
  } catch (e) {
    // silently ignore on startup
  }
}

function updateStatsGrid(data) {
  const counts = data.row_counts || {};

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl('stat-static',  counts.static_loans  !== null ? fmt(counts.static_loans)  : '—');
  setEl('stat-dynamic', counts.dynamic_performance !== null ? fmt(counts.dynamic_performance) : '—');
  setEl('stat-gold',    counts.gold_features  !== null ? fmt(counts.gold_features) : '—');
  setEl('stat-db-size', data.db_size_mb !== undefined ? fmt(data.db_size_mb, 1) : '—');

  const dr = data.date_range || {};
  if (dr.min && dr.max) {
    setEl('stat-date-range', `${dr.min} -> ${dr.max} (${dr.cutoffs} cutoffs)`);
    setEl('stat-static-sub', data.default_rate ? `${data.default_rate}% default rate` : 'Loaded');
    setEl('stat-dynamic-sub', `${dr.cutoffs} monthly cutoffs`);
    setEl('stat-gold-sub',   'ML-ready rows');
    const badge = document.getElementById('db-badge');
    if (badge) badge.style.display = 'inline-flex';
    setGlobalStatus('Online', 'status-done');
  } else if (!data.db_exists) {
    setGlobalStatus('No Database', 'status-idle');
  }
}

// -- Run pipeline stage --------------------------------------─
async function runStage(stage) {
  const btnId = `btn-${stage}`;
  setBtnLoading(btnId, true);
  setStageStatus(stage, 'status-running', 'Running');
  setProgress(stage, 10);
  clearLog(stage);
  appendLog(stage, `▶ Starting ${stage}...`);
  setGlobalStatus('Running', 'status-running');

  // Animate progress
  let prog = 10;
  const progInterval = setInterval(() => {
    prog = Math.min(prog + Math.random() * 8, 88);
    setProgress(stage, prog);
  }, 600);

  const endpoint = stage === 'validate' ? '/api/run/validate'
                 : stage === 'train'    ? '/api/run/train'
                 : stage === 'tokenize' ? '/api/run/tokenize'
                 :                        '/api/run/ingest';

  try {
    const resp = await fetch(endpoint, { method: 'POST' });
    clearInterval(progInterval);
    setProgress(stage, 100);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Request failed');
    }

    const data = await resp.json();
    const run  = data.run || {};

    // Log stdout
    if (run.stdout) {
      run.stdout.split('\n').forEach(line => {
        if (!line.trim()) return;
        const cls = line.includes('OK') || line.includes('PASS') ? 'log-success'
                  : line.includes('[FAIL]') || line.includes('FAIL') ? 'log-error'
                  : line.includes('[WARNING]')                          ? 'log-warn'
                  : '';
        appendLog(stage, line, cls);
      });
    }

    if (run.success) {
      setStageStatus(stage, 'status-done', 'Done');
      appendLog(stage, `OK Completed in ${run.elapsed_s}s`, 'log-success');
      toast(`${stage} completed successfully (${run.elapsed_s}s)`, 'success');

      // Load results into charts
      if (stage === 'validate') {
        loadValidationResults(data.report);
        switchTab('validation');
      } else if (stage === 'train') {
        loadModelResults(data.report);
        switchTab('model');
      } else if (stage === 'tokenize') {
        loadSequenceResults(data.report);
        switchTab('sequences');
      }
    } else {
      setStageStatus(stage, 'status-error', 'Error');
      appendLog(stage, run.stderr || 'Script returned non-zero exit code.', 'log-error');
      toast(`${stage} failed. Check log for details.`, 'error');
    }

    await fetchStatus();

  } catch (e) {
    clearInterval(progInterval);
    setProgress(stage, 0);
    setStageStatus(stage, 'status-error', 'Error');
    appendLog(stage, `Error: ${e.message}`, 'log-error');
    toast(e.message, 'error');
    setGlobalStatus('Error', 'status-error');
  } finally {
    setBtnLoading(btnId, false);

    // Update global status based on all stages
    const statuses = Object.values(state.stageStatus);
    if (statuses.every(s => s === 'done')) {
      setGlobalStatus('All Done', 'status-done');
    } else if (statuses.some(s => s === 'error')) {
      setGlobalStatus('Has Errors', 'status-error');
    } else if (statuses.some(s => s === 'done')) {
      setGlobalStatus('Partially Done', 'status-running');
    } else {
      setGlobalStatus('Ready', 'status-idle');
    }
  }
}

// -- Render validation checks list --------------------------─
function renderChecksList(failOnly = false) {
  const list = document.getElementById('checks-list');
  if (!list) return;

  const checks = failOnly
    ? state.validationChecks.filter(c => !c.passed)
    : state.validationChecks;

  if (checks.length === 0) {
    list.innerHTML = failOnly
      ? '<div class="empty-state"><span class="empty-icon">🎉</span><span>All checks passed!</span></div>'
      : '<div class="empty-state"><span class="empty-icon">🔍</span><span>Run Validation to see check results</span></div>';
    return;
  }

  list.innerHTML = checks.map(c => `
    <div class="check-item" title="${c.name}">
      <div class="check-icon ${c.passed ? 'check-pass' : 'check-fail'}">${c.passed ? 'OK' : '[FAIL]'}</div>
      <span class="check-name">${c.name}</span>
      <span class="check-detail">${c.detail}</span>
      <span class="check-ms">${c.elapsed_ms}ms</span>
    </div>
  `).join('');
}

// -- Load validation results ----------------------------------
function loadValidationResults(report) {
  if (!report || report.error) return;
  state.validationChecks = report.checks || [];

  // Summary numbers
  document.getElementById('checks-passed').textContent = fmt(report.passed);
  document.getElementById('checks-failed').textContent = fmt(report.failed);
  document.getElementById('checks-total').textContent  = fmt(report.total_checks);
  document.getElementById('validation-time').textContent = `${fmt(report.total_ms, 0)}ms`;

  const score = Math.round(100 * report.passed / report.total_checks);
  const scorePill = document.getElementById('validation-score');
  scorePill.textContent = `${score}%`;
  scorePill.className = `pill ${score === 100 ? 'pill-emerald' : score >= 80 ? 'pill-blue' : 'pill-violet'}`;

  renderChecksList(false);
  renderValidationCategoryChart(report.checks || []);
}

// -- Category donut chart ------------------------------------─
function renderValidationCategoryChart(checks) {
  const cats = {};
  checks.forEach(c => {
    cats[c.category] = cats[c.category] || { pass: 0, fail: 0 };
    c.passed ? cats[c.category].pass++ : cats[c.category].fail++;
  });

  const labels = Object.keys(cats);
  const passData = labels.map(k => cats[k].pass);
  const failData = labels.map(k => cats[k].fail);

  const ctx = document.getElementById('validation-cat-chart');
  if (!ctx) return;
  if (charts.validationCat) charts.validationCat.destroy();

  charts.validationCat = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Pass', data: passData, backgroundColor: 'rgba(16,185,129,0.7)', borderRadius: 4 },
        { label: 'Fail', data: failData, backgroundColor: 'rgba(244,63,94,0.6)',  borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 10, padding: 12 } } },
      scales: {
        x: { stacked: true, grid: CHART_GRID, ticks: { font: { size: 10 } } },
        y: { stacked: true, grid: CHART_GRID, ticks: { stepSize: 1 } },
      }
    }
  });
}

// -- Load model results --------------------------------------─
function loadModelResults(report) {
  if (!report || report.error) return;
  state.modelResults = report;

  // Metrics
  document.getElementById('metric-auc').textContent     = fmt(report.auc_roc_test, 4);
  document.getElementById('metric-gini').textContent    = fmt(report.gini_test, 4);
  document.getElementById('metric-avgprec').textContent = fmt(report.avg_precision, 4);
  document.getElementById('metric-defrate').textContent = fmtPct(report.test_default_rate * 100, 2);
  document.getElementById('metric-trainrows').textContent = fmt(report.train_rows);
  document.getElementById('metric-testrows').textContent  = fmt(report.test_rows);
  document.getElementById('metric-traintime').textContent = fmt(report.train_seconds, 1);
  document.getElementById('metric-split').textContent = report.split_method || '—';

  renderRocChart(report.roc_curve);
  renderTrainingCurve(report.training_curve);
  renderFeatureImportances(report.feature_importances || []);
}

// -- ROC curve chart ------------------------------------------
function renderRocChart(rocData) {
  const ctx = document.getElementById('roc-chart');
  if (!ctx || !rocData) return;
  if (charts.roc) charts.roc.destroy();

  charts.roc = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rocData.fpr,
      datasets: [
        {
          label: `ROC (AUC = ${fmt(state.modelResults?.auc_roc_test, 4)})`,
          data: rocData.tpr,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Random (AUC = 0.50)',
          data: rocData.fpr,
          borderColor: 'rgba(148,163,184,0.35)',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, padding: 12 } },
      },
      scales: {
        x: {
          title: { display: true, text: 'False Positive Rate', color: '#64748b' },
          grid: CHART_GRID,
          min: 0, max: 1,
        },
        y: {
          title: { display: true, text: 'True Positive Rate', color: '#64748b' },
          grid: CHART_GRID,
          min: 0, max: 1,
        }
      }
    }
  });
}

// -- Training curve ------------------------------------------─
function renderTrainingCurve(curveData) {
  const ctx = document.getElementById('train-curve-chart');
  if (!ctx || !curveData) return;
  if (charts.trainCurve) charts.trainCurve.destroy();

  const rounds = Array.from({ length: curveData.train_auc.length }, (_, i) => i + 1);
  charts.trainCurve = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rounds,
      datasets: [
        {
          label: 'Train AUC',
          data: curveData.train_auc,
          borderColor: '#8b5cf6',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.2,
        },
        {
          label: 'Test AUC',
          data: curveData.test_auc,
          borderColor: '#06b6d4',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.2,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, padding: 12 } },
      },
      scales: {
        x: {
          title: { display: true, text: 'Boosting Rounds', color: '#64748b' },
          grid: CHART_GRID,
          ticks: { maxTicksLimit: 10 },
        },
        y: {
          title: { display: true, text: 'AUC', color: '#64748b' },
          grid: CHART_GRID,
        }
      }
    }
  });
}

// -- Feature importances --------------------------------------
function renderFeatureImportances(features) {
  const container = document.getElementById('feature-bars');
  if (!container) return;

  document.getElementById('feature-count').textContent = `Top ${features.length}`;

  const max = features[0]?.importance || 1;
  container.innerHTML = features.map(f => `
    <div class="feature-row">
      <span class="feature-name" title="${f.feature}">${f.feature}</span>
      <div class="feature-bar-bg">
        <div class="feature-bar-fill" style="width:${(f.importance / max * 100).toFixed(1)}%"></div>
      </div>
      <span class="feature-val">${(f.importance * 100).toFixed(2)}%</span>
    </div>
  `).join('');

  renderFeatureCategoryChart(features);
}

// -- Feature category pie chart ------------------------------─
function renderFeatureCategoryChart(features) {
  const staticCols = new Set([
    'origination_year','original_balance','legal_maturity_months','oltomv_original',
    'original_market_value_at_origination','loan_to_income','payment_due_to_income_pct',
    'borrower_annual_income','debtor_count','construction_year','primary_energy_demand_kwh_m2',
    'interest_only_flag','self_employed_flag','buy_to_let_flag','nhg_flag',
    'repayment_type','rate_type','borrower_type','employment_status','loan_purpose',
    'province','economic_region_nuts3','construction_year_bucket','occupancy',
    'property_type','property_usage','epc_label',
  ]);

  let staticImp = 0, dynamicImp = 0;
  features.forEach(f => {
    if (staticCols.has(f.feature)) staticImp += f.importance;
    else dynamicImp += f.importance;
  });

  const ctx = document.getElementById('feat-category-chart');
  if (!ctx) return;
  if (charts.featCat) charts.featCat.destroy();

  charts.featCat = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Static (Origination)', 'Dynamic (Performance)'],
      datasets: [{
        data: [staticImp, dynamicImp],
        backgroundColor: ['rgba(139,92,246,0.7)', 'rgba(6,182,212,0.7)'],
        borderColor:     ['rgba(139,92,246,1)',   'rgba(6,182,212,1)'],
        borderWidth: 1.5,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${(ctx.parsed * 100).toFixed(1)}%`
          }
        }
      },
      cutout: '62%',
    }
  });
}

// -- Load sequence results ----------------------------------─
function loadSequenceResults(report) {
  if (!report || report.error) return;

  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  setEl('seq-vocab-size',    fmt(report.vocab_size));
  setEl('seq-step-width',    `${report.step_width} tokens/step`);
  setEl('seq-total-loans',   fmt(report.total_loans));
  setEl('seq-avg-len',       `Avg len ${fmt(report.avg_seq_len, 1)}`);
  setEl('seq-cure-rate',     fmtPct(report.cure_rate_pct));
  setEl('seq-cure-abs',      `${fmt(report.total_cures)} cured loans`);
  setEl('seq-default-rate',  fmtPct(report.default_rate_pct));
  setEl('seq-default-abs',   `${fmt(report.total_defaults)} defaulted loans`);

  const tm = report.transition_matrix;
  if (tm) {
    const totalTrans = Object.values(report.state_transitions || {}).reduce((a, b) => a + b, 0);
    const pill = document.getElementById('trans-total');
    if (pill) pill.textContent = `${fmt(totalTrans)} transitions`;
    renderTransitionHeatmap(tm);
  }

  renderEventFreqChart(report.event_frequency || {});
  renderVocabTiers();
  renderLoanTimelines(report);
}

// -- Render state transition heatmap ------------------------─
function renderTransitionHeatmap(tm) {
  const container = document.getElementById('transition-heatmap');
  if (!container || !tm) return;

  const states = tm.states;
  const matrix = tm.matrix;  // 2D array, row-normalized probabilities

  // Build table
  let html = '<table class="heatmap-table"><thead><tr><th>From \ To</th>';
  states.forEach(s => { html += `<th>${s}</th>`; });
  html += '</tr></thead><tbody>';

  states.forEach((fromState, i) => {
    html += `<tr><td class="row-label">${fromState}</td>`;
    states.forEach((_, j) => {
      const prob = matrix[i][j];
      const pct  = Math.round(prob * 100);
      // Color intensity: green for self-transitions (staying), amber for escalation, red for default
      let bg;
      if (i === j) {
        bg = `rgba(16,185,129,${(prob * 0.7).toFixed(2)})`;
      } else if (j > i && j < 5) {
        bg = `rgba(251,191,36,${(prob * 1.5).toFixed(2)})`;
      } else if (j >= 5) {
        bg = `rgba(239,68,68,${(prob * 2.0).toFixed(2)})`;
      } else {
        bg = `rgba(139,92,246,${(prob * 1.2).toFixed(2)})`; // cure / improvement
      }
      const txt = pct > 0 ? `${pct}%` : '·';
      html += `<td style="background:${bg};" title="${fromState}->${states[j]}: ${(prob*100).toFixed(1)}%">${txt}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

// -- Render event frequency chart ----------------------------─
function renderEventFreqChart(eventFreq) {
  const ctx = document.getElementById('event-freq-chart');
  if (!ctx) return;
  if (charts.eventFreq) charts.eventFreq.destroy();

  const COLORS = {
    PERF: '#34d399', DPD1: '#fbbf24', DPD2: '#f59e0b',
    DPD3: '#f97316', DPD4: '#ef4444', DFLT: '#dc2626',
    CHOF: '#7f1d1d', RDMD: '#a78bfa',
  };

  const labels = Object.keys(eventFreq);
  const data   = Object.values(eventFreq);
  const colors = labels.map(l => COLORS[l] || '#64748b');

  charts.eventFreq = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Observations',
        data,
        backgroundColor: colors.map(c => c + '99'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: CHART_GRID },
        y: { grid: CHART_GRID, ticks: { callback: v => fmt(v) } },
      }
    }
  });
}

// -- Render vocabulary tier breakdown ------------------------─
async function renderVocabTiers() {
  const container = document.getElementById('vocab-tiers');
  if (!container) return;

  try {
    const r = await fetch('/api/results/tokenizer');
    if (!r.ok) return;
    const data = await r.json();
    if (data.error) return;

    const tiers = data.tiers || {};
    const TIER_COLORS = {
      special:    { label: '🏷️ Special', color: '#64748b' },
      lifecycle:  { label: '🔄 Lifecycle Events', color: '#3b82f6' },
      continuous: { label: '📏 Continuous Bins', color: 'hsl(38,92%,60%)' },
    };

    let html = '';
    for (const [key, info] of Object.entries(TIER_COLORS)) {
      const tokens = tiers[key];
      if (!tokens) continue;

      let tokenKeys;
      if (key === 'continuous') {
        // Show feature names only (not all 40 individual bin tokens)
        tokenKeys = Object.keys(tokens);
        const featGroups = {};
        tokenKeys.forEach(t => {
          const feat = t.split('_Q')[0];
          featGroups[feat] = (featGroups[feat] || 0) + 1;
        });
        tokenKeys = Object.keys(featGroups).map(f => `${f}(x${featGroups[f]})`);
      } else {
        tokenKeys = Object.keys(tokens);
      }

      html += `<div class="vocab-tier">
        <span class="vocab-tier-name" style="color:${info.color}">${info.label}</span>
        <span style="color:var(--text-muted);font-size:0.72rem;">${tokenKeys.length} tokens</span>
        <div class="vocab-tier-tokens">${tokenKeys.map(t => `<span class="vocab-chip">${t}</span>`).join('')}</div>
      </div>`;
    }
    html += `<div style="text-align:right;font-size:0.72rem;color:var(--text-muted);padding-top:4px;">Total: ${data.vocab_size} tokens</div>`;
    container.innerHTML = html;
  } catch (_) {}
}

// -- Render sample loan timelines ----------------------------─
function renderLoanTimelines(report) {
  const container = document.getElementById('loan-timelines');
  if (!container || !report) return;

  // We can't stream parquet from the browser directly, so we render
  // a placeholder with state distribution for now.
  // A richer version would call a new /api/results/sample-sequences endpoint.
  const eventColors = {
    PERF:'#34d399',DPD1:'#fbbf24',DPD2:'#f59e0b',
    DPD3:'#f97316',DPD4:'#ef4444',DFLT:'#dc2626',
    CHOF:'#7f1d1d',RDMD:'#a78bfa',
  };

  // Build illustrative timeline rows from state transition distribution
  const transMap = report.state_transitions || {};
  const seqProfiles = [
    { id: 'Performing Loan',  seq: ['PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF'] },
    { id: 'Cure Path',        seq: ['PERF','PERF','DPD1','DPD2','DPD1','PERF','PERF','PERF','PERF','PERF','PERF','PERF'] },
    { id: 'Default Path',     seq: ['PERF','PERF','DPD1','DPD2','DPD3','DPD4','DFLT','DFLT','DFLT','DFLT','CHOF'] },
    { id: 'Early Prepayment', seq: ['PERF','PERF','PERF','PERF','RDMD'] },
    { id: 'Late Arrears',     seq: ['PERF','PERF','PERF','PERF','PERF','PERF','PERF','DPD1','DPD2','DPD3','DPD4','DFLT'] },
    { id: 'Quick Cure',       seq: ['PERF','DPD1','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF','PERF'] },
  ];

  container.innerHTML = seqProfiles.map(profile => {
    const steps = profile.seq.map(ev => {
      const col = eventColors[ev] || '#64748b';
      return `<div class="timeline-step" style="background:${col}22;color:${col};border:1px solid ${col}44;" title="${ev}">${ev.slice(0,2)}</div>`;
    }).join('');
    return `<div class="timeline-row">
      <div class="timeline-id">${profile.id}</div>
      <div style="display:flex;gap:3px;flex-wrap:wrap;">${steps}</div>
    </div>`;
  }).join('');
}

async function tryLoadExistingResults() {
  try {
    const valR = await fetch('/api/results/validate');
    if (valR.ok) {
      const data = await valR.json();
      if (!data.error) {
        loadValidationResults(data);
        setStageStatus('validate', 'status-done', 'Done');
        appendLog('validate', 'OK Previous validation results loaded', 'log-success');
        state.stageStatus.validate = 'done';
      }
    }
  } catch (_) {}

  try {
    const modR = await fetch('/api/results/model');
    if (modR.ok) {
      const data = await modR.json();
      if (!data.error) {
        loadModelResults(data);
        setStageStatus('train', 'status-done', 'Done');
        appendLog('train', 'OK Previous model results loaded', 'log-success');
        state.stageStatus.train = 'done';
      }
    }
  } catch (_) {}

  try {
    const ingR = await fetch('/api/results/ingest');
    if (ingR.ok) {
      const data = await ingR.json();
      if (!data.error) {
        setStageStatus('ingest', 'status-done', 'Done');
        appendLog('ingest', `OK ${fmt(data.dynamic_performance_loaded)} rows previously ingested`, 'log-success');
        state.stageStatus.ingest = 'done';
      }
    }
  } catch (_) {}

  try {
    const seqR = await fetch('/api/results/sequences');
    if (seqR.ok) {
      const data = await seqR.json();
      if (!data.error) {
        loadSequenceResults(data);
        setStageStatus('tokenize', 'status-done', 'Done');
        appendLog('tokenize', `OK ${fmt(data.total_loans)} sequences (vocab: ${fmt(data.vocab_size)} tokens)`, 'log-success');
        state.stageStatus.tokenize = 'done';
      }
    }
  } catch (_) {}
}

// =============================================================
// Step 3: Foundation Model Training
// =============================================================

let fmTrainingChart = null;
let fmRocChart = null;

async function runFoundationTraining() {
  const arch = document.getElementById('fm-arch').value;
  const strategy = document.getElementById('fm-strategy').value;
  const profile = document.getElementById('fm-profile').value;
  const preEpochs = document.getElementById('fm-pretrain-epochs').value;
  const jointEpochs = document.getElementById('fm-joint-epochs').value;
  const ftEpochs = document.getElementById('fm-finetune-epochs').value;

  setBtnLoading('btn-fm-train', true);
  const statusEl = document.getElementById('fm-train-status');
  const statusTxt = document.getElementById('fm-train-status-text');
  statusEl.className = 'status-badge status-running';
  statusTxt.textContent = 'Training...';

  clearLog('fm');
  appendLog('fm', `🚀 Starting training: ${arch} / ${strategy} / ${profile}`, 'log-success');
  setProgress('fm', 10);

  let params = new URLSearchParams({ arch, strategy, profile });
  if (preEpochs) params.set('pretrain_epochs', preEpochs);
  if (jointEpochs) params.set('joint_epochs', jointEpochs);
  if (ftEpochs) params.set('finetune_epochs', ftEpochs);

  try {
    setProgress('fm', 30);
    const resp = await fetch(`/api/run/foundation?${params}`, { method: 'POST' });
    setProgress('fm', 90);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Training failed');
    }

    const data = await resp.json();

    // Log output
    if (data.run && data.run.stdout) {
      data.run.stdout.split('\n').forEach(line => {
        const cl = line.includes('ERROR') ? 'log-error' :
                   line.includes('OK') ? 'log-success' :
                   line.includes('⚠') ? 'log-warn' : '';
        if (line.trim()) appendLog('fm', line, cl);
      });
    }
    if (data.run && data.run.stderr) {
      data.run.stderr.split('\n').forEach(line => {
        if (line.trim()) appendLog('fm', line, 'log-warn');
      });
    }

    setProgress('fm', 100);
    statusEl.className = 'status-badge status-done';
    statusTxt.textContent = `Done (${data.run?.elapsed_s || 0}s)`;

    toast(`Training complete: ${arch} / ${strategy}`, 'success');
    await loadFoundationRuns();

    // Auto-load the first result if available
    if (data.results && data.results.length > 0) {
      const runId = data.results[0].run_id;
      if (runId) await loadFoundationRunDetails(runId);
    }

  } catch (err) {
    appendLog('fm', `❌ Error: ${err.message}`, 'log-error');
    statusEl.className = 'status-badge status-error';
    statusTxt.textContent = 'Error';
    toast(`Training failed: ${err.message}`, 'error');
  } finally {
    setBtnLoading('btn-fm-train', false);
  }
}

async function loadFoundationRuns() {
  try {
    const resp = await fetch('/api/foundation/runs');
    const data = await resp.json();
    const runs = data.runs || [];

    const container = document.getElementById('fm-runs-table');

    if (runs.length === 0) {
      container.innerHTML = '<div class="empty-state"><span class="empty-icon">📜</span><span>No training runs yet</span></div>';
      return;
    }

    let html = '<div class="fm-runs-list">';
    html += `<div class="fm-run-row" style="font-weight:600; color:var(--text-muted); cursor:default; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;">
      <div>Run</div><div style="text-align:right">AUC</div><div style="text-align:right">Gini</div><div style="text-align:right">Time</div></div>`;

    for (const run of runs) {
      const archIcon = { hybrid:'🔀', patchtst:'📊', tft:'📈', lightweight:'⚡', lstm_baseline:'🔁' }[run.architecture] || '🤖';
      html += `<div class="fm-run-row" onclick="loadFoundationRunDetails('${run.run_id}')">
        <div>
          <div class="fm-run-name">${archIcon} ${run.architecture}</div>
          <div class="fm-run-arch">${run.strategy} · ${run.profile}</div>
        </div>
        <div class="fm-run-metric accent-blue">${run.auc_roc_default ? run.auc_roc_default.toFixed(4) : '—'}</div>
        <div class="fm-run-metric accent-violet">${run.gini_default ? run.gini_default.toFixed(4) : '—'}</div>
        <div class="fm-run-metric" style="color:var(--text-muted)">${run.total_time_s ? run.total_time_s.toFixed(0) + 's' : '—'}</div>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;

    if (typeof populateDownstreamRunSelect === 'function') {
      populateDownstreamRunSelect(runs);
    }

  } catch (err) {
    console.error('Failed to load runs:', err);
  }
}

async function loadFoundationRunDetails(runId) {
  try {
    const resp = await fetch(`/api/foundation/run/${encodeURIComponent(runId)}`);
    const results = await resp.json();

    // Update selected run label
    document.getElementById('fm-selected-run-id').textContent = runId.split('_').slice(0, 2).join(' ');

    // Highlight selected row
    document.querySelectorAll('.fm-run-row').forEach(r => r.classList.remove('selected'));

    // Build metrics display
    const metricsDiv = document.getElementById('fm-run-metrics');
    const stages = results.stages || {};

    let html = '';

    // Finetune metrics (most important)
    const ft = stages.finetune?.metrics || {};
    const def = ft.default || {};
    const cure = ft.cure || {};

    if (def.auc_roc) {
      html += '<div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">Default Prediction</div>';
      html += '<div class="fm-metrics-grid" style="margin-bottom:16px;">';
      html += fmMetricCard(def.auc_roc, 'AUC-ROC', 'accent-blue');
      html += fmMetricCard(def.gini, 'Gini', 'accent-violet');
      html += fmMetricCard(def.ks_statistic, 'KS', 'accent-cyan');
      html += fmMetricCard(def.avg_precision, 'Avg Prec', 'accent-emerald');
      html += fmMetricCard(def.f1_score, 'F1', 'accent-amber');
      html += fmMetricCard(def.brier_score, 'Brier', 'accent-rose');
      html += '</div>';
    }

    if (cure.auc_roc) {
      html += '<div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">Cure Prediction</div>';
      html += '<div class="fm-metrics-grid">';
      html += fmMetricCard(cure.auc_roc, 'AUC-ROC', 'accent-blue');
      html += fmMetricCard(cure.gini, 'Gini', 'accent-violet');
      html += fmMetricCard(cure.f1_score, 'F1', 'accent-amber');
      html += '</div>';
    }

    // Pretrain metrics
    const pt = stages.pretrain?.metrics || {};
    if (pt.masked_accuracy !== undefined) {
      html += '<div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin: 16px 0 8px;">Pretrain (Self-Supervised)</div>';
      html += '<div class="fm-metrics-grid">';
      html += fmMetricCard(pt.masked_accuracy, 'Mask Acc', 'accent-blue');
      html += fmMetricCard(pt.top3_accuracy, 'Top-3 Acc', 'accent-cyan');
      html += fmMetricCard(pt.perplexity, 'Perplexity', 'accent-violet');
      html += '</div>';
    }

    // Summary
    html += `<div style="margin-top:16px; padding-top:12px; border-top:1px solid var(--border-subtle); font-size:0.72rem; color:var(--text-muted); display:flex; justify-content:space-between;">
      <span>Params: <span class="accent-cyan">${(results.total_params || 0).toLocaleString()}</span></span>
      <span>Time: <span class="accent-amber">${(results.total_time_s || 0).toFixed(1)}s</span></span>
    </div>`;

    metricsDiv.innerHTML = html || '<div class="empty-state">No metrics available</div>';

    // Render charts
    renderFmTrainingChart(stages);
    renderFmRocChart(def);

  } catch (err) {
    console.error('Failed to load run details:', err);
  }
}

function fmMetricCard(value, label, colorClass) {
  const display = value !== undefined && value !== null ? Number(value).toFixed(4) : '—';
  return `<div class="fm-metric-card">
    <div class="fm-metric-value ${colorClass}">${display}</div>
    <div class="fm-metric-label">${label}</div>
  </div>`;
}

function renderFmTrainingChart(stages) {
  const canvas = document.getElementById('fm-training-chart');
  if (!canvas) return;

  if (fmTrainingChart) { fmTrainingChart.destroy(); fmTrainingChart = null; }

  // Collect training curves from all stages
  const datasets = [];
  const colors = {
    pretrain: { train: '#3b82f6', val: '#93c5fd' },
    joint: { train: '#8b5cf6', val: '#c4b5fd' },
    finetune: { train: '#10b981', val: '#6ee7b7' },
  };

  let maxEpochs = 0;
  for (const [stageName, stageData] of Object.entries(stages)) {
    if (!stageData.train_losses) continue;
    const n = stageData.train_losses.length;
    maxEpochs = Math.max(maxEpochs, n);

    datasets.push({
      label: `${stageName} train`,
      data: stageData.train_losses,
      borderColor: colors[stageName]?.train || '#94a3b8',
      borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0,
    });
    if (stageData.val_losses) {
      datasets.push({
        label: `${stageName} val`,
        data: stageData.val_losses,
        borderColor: colors[stageName]?.val || '#cbd5e1',
        borderWidth: 1.5, borderDash: [4, 3], fill: false, tension: 0.3, pointRadius: 0,
      });
    }
  }

  if (datasets.length === 0) return;

  fmTrainingChart = new Chart(canvas, {
    type: 'line',
    data: { labels: Array.from({length: maxEpochs}, (_, i) => i + 1), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 8, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'Epoch' }, grid: CHART_GRID },
        y: { title: { display: true, text: 'Loss' }, grid: CHART_GRID },
      },
    },
  });
}

function renderFmRocChart(defaultMetrics) {
  const canvas = document.getElementById('fm-roc-chart');
  if (!canvas) return;

  if (fmRocChart) { fmRocChart.destroy(); fmRocChart = null; }

  const roc = defaultMetrics?.roc_curve;
  if (!roc || !roc.fpr || !roc.tpr) return;

  fmRocChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: roc.fpr,
      datasets: [
        {
          label: `ROC (AUC=${(defaultMetrics.auc_roc || 0).toFixed(4)})`,
          data: roc.tpr,
          borderColor: '#3b82f6', borderWidth: 2.5, fill: false, tension: 0.2, pointRadius: 0,
        },
        {
          label: 'Random',
          data: roc.fpr,
          borderColor: 'rgba(148,163,184,0.3)', borderWidth: 1, borderDash: [6, 4],
          fill: false, pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 8, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'FPR' }, min: 0, max: 1, grid: CHART_GRID },
        y: { title: { display: true, text: 'TPR' }, min: 0, max: 1, grid: CHART_GRID },
      },
    },
  });
}

async function loadFoundationComparison() {
  try {
    const resp = await fetch('/api/foundation/compare');
    const data = await resp.json();
    const runs = data.runs || [];

    const card = document.getElementById('fm-comparison-card');
    const container = document.getElementById('fm-comparison-table');

    if (runs.length === 0) {
      card.style.display = 'none';
      toast('No runs to compare', 'info');
      return;
    }

    card.style.display = 'block';

    // Find best values for highlighting
    const bestAuc = Math.max(...runs.map(r => r.auc_roc_default || 0));
    const bestGini = Math.max(...runs.map(r => r.gini_default || 0));
    const bestKS = Math.max(...runs.map(r => r.ks_statistic || 0));

    let html = '<table class="fm-comp-table"><thead><tr>';
    html += '<th>Architecture</th><th>Strategy</th><th>AUC-ROC</th><th>Gini</th><th>AP</th><th>KS</th><th>Brier</th><th>Params</th><th>Time</th>';
    html += '</tr></thead><tbody>';

    for (const run of runs) {
      html += '<tr>';
      html += `<td style="font-weight:600; color:var(--text-primary)">${run.architecture}</td>`;
      html += `<td>${run.strategy}</td>`;
      html += `<td class="${run.auc_roc_default === bestAuc ? 'best' : ''}">${(run.auc_roc_default || 0).toFixed(4)}</td>`;
      html += `<td class="${run.gini_default === bestGini ? 'best' : ''}">${(run.gini_default || 0).toFixed(4)}</td>`;
      html += `<td>${(run.avg_precision || 0).toFixed(4)}</td>`;
      html += `<td class="${run.ks_statistic === bestKS ? 'best' : ''}">${(run.ks_statistic || 0).toFixed(4)}</td>`;
      html += `<td>${(run.brier_score || 0).toFixed(6)}</td>`;
      html += `<td>${(run.params || 0).toLocaleString()}</td>`;
      html += `<td>${(run.total_time_s || 0).toFixed(0)}s</td>`;
      html += '</tr>';
    }

    // XGBoost baseline row
    if (data.baseline) {
      html += '<tr class="baseline-row">';
      html += '<td style="font-weight:700;">XGBoost Baseline</td>';
      html += '<td>—</td>';
      html += `<td>${(data.baseline.auc_roc || 0).toFixed(4)}</td>`;
      html += `<td>${(data.baseline.gini || 0).toFixed(4)}</td>`;
      html += '<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>';
      html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    toast('Comparison loaded', 'success');

  } catch (err) {
    console.error('Failed to load comparison:', err);
    toast('Failed to load comparison', 'error');
  }
}

// =============================================================
// Step 4: Downstream Evaluation
// =============================================================

function populateDownstreamRunSelect(runs) {
  const select = document.getElementById('downstream-run-select');
  if (!select) return;
  select.innerHTML = '<option value="">-- Select Trained Run --</option>';
  runs.forEach(r => {
    if (r.has_embeddings) {
      const opt = document.createElement('option');
      opt.value = r.run_id;
      opt.textContent = `${r.architecture} · ${r.strategy} (${r.run_id.substring(r.run_id.length - 6)})`;
      select.appendChild(opt);
    }
  });
}

async function triggerDownstreamEval() {
  const runId = document.getElementById('downstream-run-select').value;
  if (!runId) {
    toast('Please select a run first', 'warn');
    return;
  }
  
  setBtnLoading('btn-run-downstream', true);
  document.getElementById('downstream-status-message').textContent = 'Running downstream evaluation... (this takes a minute)';
  
  try {
    const resp = await fetch(`/api/run/downstream/${encodeURIComponent(runId)}`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Downstream evaluation failed');
    }
    const data = await resp.json();
    toast('Downstream evaluation complete!', 'success');
    document.getElementById('downstream-status-message').textContent = 'Success!';
    renderDownstreamResults(data.results);
  } catch (err) {
    toast(`Downstream error: ${err.message}`, 'error');
    document.getElementById('downstream-status-message').textContent = `Error: ${err.message}`;
  } finally {
    setBtnLoading('btn-run-downstream', false);
  }
}

function renderDownstreamResults(results) {
  if (!results || !results.baseline) return;
  
  document.getElementById('downstream-charts-grid').style.display = 'grid';
  
  // Render Leaderboard
  const models = [
    { id: 'hybrid', name: 'Hybrid (Features + Embeddings)', res: results.hybrid },
    { id: 'embeddings', name: 'Embeddings Only (XGBoost)', res: results.embeddings },
    { id: 'dnn', name: 'PyTorch DNN (Embeddings)', res: results.dnn },
    { id: 'linear', name: 'Linear Probe (Logistic)', res: results.linear_probe },
    { id: 'baseline', name: 'Baseline (Handcrafted Features)', res: results.baseline },
  ].filter(m => m.res);
  
  // Sort by PR-AUC descending
  models.sort((a, b) => b.res.pr_auc - a.res.pr_auc);
  
  const tbody = document.querySelector('#downstream-leaderboard-table tbody');
  let html = '';
  const bestPrAuc = models[0].res.pr_auc;
  
  models.forEach(m => {
    html += `<tr>
      <td style="font-weight:600; color:var(--text-primary)">${m.name}</td>
      <td class="${m.res.pr_auc === bestPrAuc ? 'best' : ''}">${m.res.pr_auc.toFixed(4)}</td>
      <td>${m.res.auc_roc.toFixed(4)}</td>
      <td>${m.res.gini.toFixed(4)}</td>
      <td>${m.res.brier_score.toFixed(4)}</td>
      <td>${m.res.ece.toFixed(4)}</td>
      <td>${(m.res.capture_rate_5pct * 100).toFixed(1)}%</td>
    </tr>`;
  });
  tbody.innerHTML = html;
  
  // Business Metrics
  const baseCapture = results.baseline.capture_rate_5pct;
  const bestCapture = results.hybrid ? results.hybrid.capture_rate_5pct : models[0].res.capture_rate_5pct;
  const lift = baseCapture > 0 ? (bestCapture / baseCapture) : 1;
  document.getElementById('business-outreach-lift').textContent = lift.toFixed(2) + 'x';
  document.getElementById('business-capture-rate').textContent = (bestCapture * 100).toFixed(1) + '%';
  
  renderDownstreamEarlyWarning(models);
  renderDownstreamCalibration(models);
  renderDownstreamCohort(results.esg_slices || {});
}

let dsWarningChart = null;
let dsCalibChart = null;
let dsCohortChart = null;

function renderDownstreamEarlyWarning(models) {
  const canvas = document.getElementById('downstream-early-warning-chart');
  if (!canvas) return;
  if (dsWarningChart) dsWarningChart.destroy();
  
  const colors = {
    'hybrid': '#10b981', 'embeddings': '#8b5cf6', 'dnn': '#0ea5e9', 'linear': '#f59e0b', 'baseline': '#64748b'
  };
  
  const datasets = models.map(m => {
    const horizons = m.res.horizon_curve || [];
    horizons.sort((a, b) => a.horizon - b.horizon);
    return {
      label: m.name,
      data: horizons.map(h => h.pr_auc),
      borderColor: colors[m.id] || '#cbd5e1',
      borderWidth: m.id === 'hybrid' ? 3 : 2,
      borderDash: m.id === 'baseline' ? [5, 5] : [],
      tension: 0.2, fill: false
    };
  });
  
  dsWarningChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: [1, 2, 3, 4, 5, 6],
      datasets
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 8, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'Months to Default' }, grid: CHART_GRID },
        y: { title: { display: true, text: 'PR-AUC' }, grid: CHART_GRID, min: 0 }
      }
    }
  });
}

function renderDownstreamCalibration(models) {
  const canvas = document.getElementById('downstream-calibration-chart');
  if (!canvas) return;
  if (dsCalibChart) dsCalibChart.destroy();
  
  const colors = {
    'hybrid': '#10b981', 'baseline': '#64748b'
  };
  
  const datasets = [];
  
  // Ideal line
  datasets.push({
    label: 'Perfect Calibration',
    data: [{x:0, y:0}, {x:1, y:1}],
    borderColor: 'rgba(148,163,184,0.3)',
    borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false
  });
  
  models.filter(m => m.id === 'hybrid' || m.id === 'baseline').forEach(m => {
    const calib = m.res.calibration || [];
    datasets.push({
      label: m.name,
      data: calib.map(c => ({ x: c.mean_prob, y: c.fraction_pos })),
      borderColor: colors[m.id],
      backgroundColor: colors[m.id],
      borderWidth: 2,
      showLine: true,
      tension: 0.1
    });
  });
  
  dsCalibChart = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 8, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'Mean Predicted Probability' }, grid: CHART_GRID, min: 0, max: 0.2 },
        y: { title: { display: true, text: 'Fraction of Positives' }, grid: CHART_GRID, min: 0, max: 0.2 }
      }
    }
  });
}

function renderDownstreamCohort(esgSlices) {
  const canvas = document.getElementById('downstream-cohort-chart');
  if (!canvas) return;
  if (dsCohortChart) dsCohortChart.destroy();
  
  const labels = Object.keys(esgSlices).sort();
  const data = labels.map(k => esgSlices[k]);
  
  dsCohortChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Hybrid PR-AUC by EPC Label',
        data,
        backgroundColor: '#10b98199',
        borderColor: '#10b981',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'EPC Rating (Encoded)' }, grid: CHART_GRID },
        y: { title: { display: true, text: 'PR-AUC' }, grid: CHART_GRID, min: 0 }
      }
    }
  });
}


// -- Init ----------------------------------------------------─
(async function init() {
  setGlobalStatus('Loading', 'status-running');
  await fetchStatus();
  await tryLoadExistingResults();

  // Load foundation runs
  try { await loadFoundationRuns(); } catch (_) {}

  const allDone = Object.values(state.stageStatus).every(s => s === 'done');
  if (allDone) {
    setGlobalStatus('All Done', 'status-done');
  } else {
    setGlobalStatus('Ready', 'status-idle');
  }

  // Auto-refresh DB stats every 30s
  setInterval(fetchStatus, 30_000);
})();

