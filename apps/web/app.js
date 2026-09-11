const API = window.location.origin;
const $ = (id) => document.getElementById(id);
const state = {
  queue: [],
  selectedReport: null,
  analysisContext: "new",
  analyticsExpanded: false,
};

function escapeHtml(v) {
  return String(v ?? "").replace(
    /[&<>'"]/g,
    (m) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#039;",
        '"': "&quot;",
      })[m],
  );
}
function toast(msg, kind = "info") {
  const t = $("toast");
  t.textContent = msg;
  t.dataset.kind = kind;
  t.classList.add("show");
  clearTimeout(window.__toast);
  window.__toast = setTimeout(() => t.classList.remove("show"), 2800);
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(API + path, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
      ...options,
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }
    if (!response.ok)
      throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  } catch (err) {
    if (err.name === "AbortError")
      throw new Error(
        "Request timed out. Check that the RAKSHAK API is running.",
      );
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
function pill(priority) {
  return `<span class="pill ${escapeHtml(String(priority || "P5").toLowerCase())}">${escapeHtml(priority || "P5")}</span>`;
}
function decisionPill(decision) {
  const d = decision || "UNREVIEWED";
  return `<span class="decision ${d.toLowerCase()}">${escapeHtml(d.replaceAll("_", " "))}</span>`;
}
function switchTab(name) {
  document
    .querySelectorAll(".tab")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document
    .querySelectorAll(".panel")
    .forEach((p) => p.classList.toggle("active", p.id === name));
  if (name !== "analyze") state.selectedReport = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "overview") loadOverview();
  if (name === "queue") loadQueue();
  if (name === "analytics") loadAnalytics();
}
document.querySelectorAll("[data-tab]").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.dataset.tab === "analyze") startNewAnalysis();
    else switchTab(b.dataset.tab);
  }),
);

async function loadHealth() {
  try {
    const d = await api("/health");
    $("statusDot").classList.add("ok");
    $("statusText").textContent = `API connected · ${d.device || "cpu"}`;
  } catch {
    $("statusDot").classList.remove("ok");
    $("statusText").textContent = "API unavailable";
  }
}
async function loadOverview() {
  try {
    const [d, reviews] = await Promise.all([
      api("/analytics/overview"),
      api("/review-stats"),
    ]);
    $("overviewCards").innerHTML = [
      ["Records", d.records, "Prediction set"],
      ["AI flags ≥ 0.50", d.model_yes_ge_0_50, "Model triage signal"],
      ["High score ≥ 0.75", d.high_ge_0_75, "Higher review attention"],
      ["Human reviews", reviews.total_reviews ?? 0, "Persisted decisions"],
    ]
      .map(
        (x) =>
          `<div class="stat"><div class="k">${escapeHtml(x[0])}</div><div class="v">${escapeHtml(x[1])}</div><div class="s">${escapeHtml(x[2])}</div></div>`,
      )
      .join("");
    const dist = d.priority_distribution || {};
    const max = Math.max(...Object.values(dist), 1);
    $("priorityBars").innerHTML = ["P1", "P2", "P3", "P4", "P5"]
      .map(
        (k) =>
          `<div class="bar-row"><strong>${k}</strong><div class="bar-track"><div class="bar-fill" style="width:${(Number(dist[k] || 0) / max) * 100}%"></div></div><strong>${Number(dist[k] || 0)}</strong></div>`,
      )
      .join("");
    $("posture").innerHTML = [
      ["Model", "RAKSHAK final v1.2"],
      ["Reviewer policy", "v1.4"],
      ["Evidence engine", "v1.5"],
      ["Activity metadata", "v0.5"],
      ["Device", d.device || "unknown"],
      ["Human review", "Mandatory"],
    ]
      .map(
        (x) =>
          `<div class="fact"><span>${escapeHtml(x[0])}</span><strong>${escapeHtml(x[1])}</strong></div>`,
      )
      .join("");
  } catch (e) {
    toast(e.message, "error");
  }
}
async function loadQueue() {
  const error = $("queueError");
  error.classList.add("hidden");
  try {
    const priority = $("priorityFilter").value;
    const d = await api(
      `/queue?limit=500${priority ? `&priority=${encodeURIComponent(priority)}` : ""}`,
    );
    state.queue = d.items || [];
    renderQueue();
  } catch (e) {
    $("queueBody").innerHTML = "";
    error.textContent = e.message;
    error.classList.remove("hidden");
    toast(e.message, "error");
  }
}
function renderQueue() {
  const search = $("queueSearch").value.trim().toLowerCase();
  const status = $("statusFilter").value;
  const rows = state.queue.filter((x) => {
    const hay = [
      x.report_id,
      x.site,
      x.activity,
      x.location,
      x.department,
      x.description,
    ]
      .map((v) => String(v ?? "").toLowerCase())
      .join(" ");
    return (
      (!search || hay.includes(search)) &&
      (!status || String(x.review_status ?? "") === status)
    );
  });
  $("queueCount").textContent =
    `${rows.length} matching of ${state.queue.length} reports`;
  $("queueFilterHint").textContent =
    search || status || $("priorityFilter").value
      ? "Active filters are applied to the loaded queue."
      : "Search across report, site, activity, location, department and narrative.";
  $("clearQueueSearch").disabled = !search;
  $("queueBody").innerHTML = rows
    .map(
      (x) =>
        `<tr><td class="report-cell"><strong>${escapeHtml(x.report_id)}</strong><div class="muted clamp">${escapeHtml(x.description || "")}</div></td><td>${pill(x.review_priority)}</td><td class="score">${(Number(x.sif_precursor_probability || 0) * 100).toFixed(1)}%</td><td>${escapeHtml(x.activity || "—")}</td><td>${escapeHtml(x.site || "—")}</td><td>${decisionPill(x.review_status)}</td><td><button class="ghost view-btn" data-id="${escapeHtml(x.report_id)}">View</button></td></tr>`,
    )
    .join("");
  $("queueEmpty").classList.toggle("hidden", rows.length !== 0);
  $("queueBody")
    .querySelectorAll(".view-btn")
    .forEach((btn) =>
      btn.addEventListener("click", () => openReport(btn.dataset.id)),
    );
}
async function openReport(reportId) {
  try {
    const d = await api(`/reports/${encodeURIComponent(reportId)}`);
    state.selectedReport = d;
    state.analysisContext = "existing";
    $("reportId").value = d.report_id || "";
    $("description").value = d.description || "";
    updateDescriptionCount();
    updateAnalysisContextUI();
    renderPrediction(d);
    switchTab("analyze");
  } catch (e) {
    toast(e.message, "error");
  }
}
function startNewAnalysis() {
  state.selectedReport = null;
  state.analysisContext = "new";
  $("reportId").value = "";
  $("description").value = "";
  $("prediction").classList.add("hidden");
  $("prediction").innerHTML = "";
  updateDescriptionCount();
  updateAnalysisContextUI();
  switchTab("analyze");
  setTimeout(() => $("description").focus(), 50);
  toast("Previous report data is cleared.", "success");
}
function updateAnalysisContextUI() {
  const existing = state.analysisContext === "existing";
  $("analysisMode").textContent = existing
    ? "EXISTING REPORT"
    : "NEW NARRATIVE";
  $("analysisContextText").textContent = existing
    ? "Inspect, analyze and review this persisted queue report."
    : "Previous report data is cleared. Enter a new narrative to begin.";
}
function updateDescriptionCount() {
  const n = $("description").value.length;
  $("descriptionCount").textContent =
    `${n.toLocaleString()} / 20,000 characters`;
}

function safeNode(id) {
  return document.getElementById(id);
}
function setHTML(id, html) {
  const el = safeNode(id);
  if (el) el.innerHTML = html;
  return el;
}
function setText(id, text) {
  const el = safeNode(id);
  if (el) el.textContent = text;
  return el;
}
function firstNum(obj, keys, fallback = 0) {
  for (const key of keys) {
    const value = Number(obj?.[key]);
    if (Number.isFinite(value)) return value;
  }
  return fallback;
}
function normalizeItems(payload) {
  return Array.isArray(payload?.items) ? payload.items.filter(Boolean) : [];
}
function normalizeCountItems(items) {
  return (Array.isArray(items) ? items : [])
    .map((x) => ({
      category: String(x?.category ?? "Unknown"),
      count: Number(x?.count ?? 0) || 0,
    }))
    .filter((x) => x.count >= 0);
}
function normalizeRateItems(items) {
  return (Array.isArray(items) ? items : []).map((x) => ({
    category: String(x?.category ?? "Unknown"),
    records: Number(x?.records ?? 0) || 0,
    model_flags: Number(x?.model_flags ?? 0) || 0,
    model_flag_rate: Math.max(
      0,
      Math.min(1, Number(x?.model_flag_rate ?? 0) || 0),
    ),
  }));
}

async function loadAnalytics() {
  const status = safeNode("analyticsStatus");
  if (status) status.textContent = "Loading analytics…";
  const endpoints = {
    overview: "/analytics/overview",
    hazards: "/analytics/hazards?limit=50",
    exposures: "/analytics/exposures?limit=50",
    pathways: "/analytics/pathways?limit=50",
    locations: "/analytics/locations?limit=50",
    activities: "/analytics/activities?limit=50",
    lsr: "/analytics/lsr?limit=50",
  };
  const entries = Object.entries(endpoints);
  const settled = await Promise.allSettled(entries.map(([, url]) => api(url)));
  const data = {};
  const failures = [];
  entries.forEach(([name], i) => {
    const result = settled[i];
    if (result.status === "fulfilled") data[name] = result.value || {};
    else {
      data[name] = {};
      failures.push(name);
    }
  });

  const overview = data.overview || {};
  const hazards = normalizeCountItems(normalizeItems(data.hazards));
  const exposures = normalizeCountItems(normalizeItems(data.exposures));
  const pathways = normalizeCountItems(normalizeItems(data.pathways));
  const lsr = normalizeCountItems(normalizeItems(data.lsr));
  const activities = normalizeRateItems(normalizeItems(data.activities));
  const locations = normalizeRateItems(normalizeItems(data.locations));
  state.analyticsData = {
    overview,
    hazards,
    exposures,
    pathways,
    lsr,
    activities,
    locations,
  };

  renderAnalyticsSummary(overview);
  renderPriorityChart(overview.priority_distribution || {});
  renderSignalFunnel(overview, exposures, pathways);
  renderHorizontalChart("hazardsChart", hazards, "count", 8);
  renderHorizontalChart("exposuresChart", exposures, "count", 8);
  renderHorizontalChart("pathwaysChart", pathways, "count", 10);
  renderHorizontalChart("lsrChart", lsr, "count", 8);
  renderHorizontalChart(
    "activitiesChart",
    activities,
    "model_flags",
    8,
    "records",
  );
  renderHotspots("locationsChart", locations, 10);

  // Keep legacy detail containers populated for compatibility, but not visible in the new UI.
  renderCountDetails("hazardsDetails", hazards);
  renderCountDetails("exposuresDetails", exposures);
  renderCountDetails("pathwaysDetails", pathways);
  renderCountDetails("lsrDetails", lsr);
  renderDensityDetails("activitiesDetails", activities);
  renderDensityDetails("locationsDetails", locations);

  const recordCount = firstNum(overview, ["records"]);
  if (status) {
    status.textContent = failures.length
      ? `${recordCount.toLocaleString()} records · ${failures.length} view${failures.length === 1 ? "" : "s"} unavailable`
      : `${recordCount.toLocaleString()} records · refreshed now`;
    status.dataset.state = failures.length ? "partial" : "ready";
  }
  if (failures.length && failures.length === entries.length)
    toast("Analytics data could not be loaded. Check the API.", "error");
  else if (failures.length)
    toast(
      `Analytics loaded with ${failures.length} unavailable view${failures.length === 1 ? "" : "s"}.`,
      "error",
    );
}

function renderAnalyticsSummary(overview) {
  const records = firstNum(overview, ["records"]);
  const flags = firstNum(overview, [
    "model_yes_ge_0_50",
    "yes",
    "model_yes",
    "predicted_yes",
  ]);
  const high = firstNum(overview, [
    "high_ge_0_75",
    "high",
    "very_high_ge_0_90",
  ]);
  const complete = firstNum(overview, [
    "complete_pathways",
    "complete_pathway_count",
  ]);
  const cards = [
    ["Records analysed", records.toLocaleString(), "Current prediction set"],
    [
      "Model-flagged",
      flags.toLocaleString(),
      `${records ? ((flags / records) * 100).toFixed(0) : 0}% of records`,
    ],
    ["High-score cases", high.toLocaleString(), "Probability ≥ 0.75"],
    [
      "Complete pathways",
      complete.toLocaleString(),
      "Hazard + exposure + mechanism",
    ],
  ];
  setHTML(
    "analyticsSummary",
    cards
      .map(
        ([label, value, sub]) =>
          `<div class="analytics-summary-card kpi-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(sub)}</small></div>`,
      )
      .join(""),
  );
}
function renderPriorityChart(dist) {
  const items = ["P1", "P2", "P3", "P4", "P5"].map((k) => ({
    category: k,
    count: Number(dist?.[k] ?? 0) || 0,
  }));
  const max = Math.max(1, ...items.map((x) => x.count));
  setHTML(
    "priorityChart",
    items
      .map(
        (x) =>
          `<div class="priority-row"><div class="priority-label"><strong>${x.category}</strong><span>${x.count.toLocaleString()}</span></div><div class="priority-track"><div class="priority-fill ${x.category.toLowerCase()}" style="width:${(x.count / max) * 100}%"></div></div></div>`,
      )
      .join(""),
  );
}
function renderSignalFunnel(overview, exposures, pathways) {
  const records = Math.max(1, firstNum(overview, ["records"]));
  const hazardPct = Number(overview?.hazard_signal_pct);
  const exposureRecords = exposures.reduce((n, x) => n + x.count, 0);
  const pathwayRecords = pathways.reduce((n, x) => n + x.count, 0);
  const complete = firstNum(overview, [
    "complete_pathways",
    "complete_pathway_count",
  ]);
  const items = [
    ["Hazard signal", Number.isFinite(hazardPct) ? hazardPct : 0],
    ["Exposure signal", Math.min(100, (exposureRecords / records) * 100)],
    ["Complete pathway", Math.min(100, (complete / records) * 100)],
  ];
  setHTML(
    "signalFunnel",
    items
      .map(
        ([label, pct]) =>
          `<div class="funnel-row"><div><strong>${escapeHtml(label)}</strong><span>${Number(pct).toFixed(0)}%</span></div><div class="funnel-track"><div style="width:${Math.max(0, Math.min(100, Number(pct) || 0))}%"></div></div></div>`,
      )
      .join(""),
  );
}
function renderHorizontalChart(
  id,
  items,
  metric = "count",
  limit = 8,
  secondary = "",
) {
  const el = safeNode(id);
  if (!el) return;
  const rows = (Array.isArray(items) ? items : []).slice(0, limit);
  if (!rows.length) {
    el.innerHTML = `<div class="analytics-empty">No data available.</div>`;
    return;
  }
  const max = Math.max(
    1,
    ...rows.map((x) => Number(x?.[metric] ?? x?.count ?? 0) || 0),
  );
  el.innerHTML = rows
    .map((x, i) => {
      const value = Number(x?.[metric] ?? x?.count ?? 0) || 0;
      const sub = secondary
        ? `${Number(x?.[secondary] ?? 0).toLocaleString()} records`
        : metric === "count"
          ? `${value.toLocaleString()} records`
          : `${value.toLocaleString()} flags`;
      return `<div class="hbar-row"><div class="hbar-meta"><span class="hbar-rank">${i + 1}</span><span class="hbar-label" title="${escapeHtml(x.category)}">${escapeHtml(x.category)}</span><strong>${value.toLocaleString()}</strong></div><div class="hbar-track"><div class="hbar-fill" style="width:${(value / max) * 100}%"></div></div><span class="hbar-sub">${escapeHtml(sub)}</span></div>`;
    })
    .join("");
}
function renderHotspots(id, items, limit = 10) {
  const el = safeNode(id);
  if (!el) return;
  const rows = (Array.isArray(items) ? items : [])
    .slice()
    .sort((a, b) => b.model_flag_rate - a.model_flag_rate)
    .slice(0, limit);
  if (!rows.length) {
    el.innerHTML = `<div class="analytics-empty">No location data available.</div>`;
    return;
  }
  el.innerHTML = rows
    .map(
      (x, i) =>
        `<div class="hotspot-row"><div class="hotspot-rank">${i + 1}</div><div class="hotspot-main"><div class="hotspot-title"><strong>${escapeHtml(x.category)}</strong><span>${(x.model_flag_rate * 100).toFixed(1)}% flag rate</span></div><div class="hotspot-track"><div style="width:${x.model_flag_rate * 100}%"></div></div></div><div class="hotspot-meta"><strong>${x.records.toLocaleString()}</strong><span>${x.model_flags.toLocaleString()} flags</span></div></div>`,
    )
    .join("");
}
function renderCountDetails(id, items) {
  const el = safeNode(id);
  if (!el) return;
  el.innerHTML = (items || [])
    .map(
      (x, i) =>
        `<div class="analytics-detail-row"><span>${i + 1}. ${escapeHtml(x.category)}</span><strong>${Number(x.count || 0).toLocaleString()}</strong></div>`,
    )
    .join("");
}
function renderDensityDetails(id, items) {
  const el = safeNode(id);
  if (!el) return;
  el.innerHTML = (items || [])
    .map(
      (x, i) =>
        `<div class="analytics-detail-row"><span>${i + 1}. ${escapeHtml(x.category)}</span><strong>${x.records.toLocaleString()} records · ${x.model_flags.toLocaleString()} flags · ${(x.model_flag_rate * 100).toFixed(1)}% flag rate</strong></div>`,
    )
    .join("");
}
function syncAnalyticsExpanded() {
  /* legacy no-op: analytics is permanently visualized */
}

function metric(label, value) {
  return `<div class="result-box"><span>${escapeHtml(label)}</span><strong>${Number(value || 0).toFixed(2)}</strong></div>`;
}
function evidenceRow(label, value) {
  return `<div class="evidence-row"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value || "None detected")}</span></div>`;
}
function renderPrediction(d) {
  const e = d.evidence || {};
  const reviews = d.reviews || [];
  const candidates = d.lsr_candidates || [];
  const priority = d.review?.priority || "P3";
  $("prediction").classList.remove("hidden");
  const reviewAllowed = state.analysisContext === "existing" && d.report_id;
  $("prediction").innerHTML =
    `<article class="card result-card"><div class="result-head"><div><p class="eyebrow">AI TRIAGE RESULT</p><div class="result-id">${escapeHtml(d.report_id || "New narrative")}</div><div class="big-score">${(Number(d.sif_precursor_probability || 0) * 100).toFixed(1)}%</div><div class="muted">SIF precursor triage probability</div></div><div class="priority-block">${pill(priority)}<div class="reason">${escapeHtml(d.review?.reason || "")}</div></div></div><div class="result-grid">${metric("Hazard signal", e.hazard_signal)}${metric("Exposure signal", e.exposure_signal)}${metric("Pathway signal", e.pathway_signal)}</div><div class="detail-grid"><section class="subcard"><div class="subhead"><h3>Evidence</h3><span class="muted">Narrative-derived</span></div>${evidenceRow("Hazards", e.hazards)}${evidenceRow("Exposures", e.exposures)}${evidenceRow("Pathways", e.pathways)}${evidenceRow("Context", e.sif_context_status)}${evidenceRow("Assessment", e.pathway_assessment)}</section><section class="subcard"><div class="subhead"><h3>Life-Saving Rule candidates</h3><span class="muted">Heuristic mapping</span></div>${candidates.length ? candidates.map((c) => `<div class="lsr-row"><div><strong>${escapeHtml(c.rule)}</strong><div class="muted">${escapeHtml((c.matched_terms || []).join(", "))}</div></div><span>${(Number(c.candidate_score || 0) * 100).toFixed(0)}%</span></div>`).join("") : '<div class="muted">No candidate rule identified.</div>'}</section></div>${reviewAllowed ? renderReviewPanel(d, reviews) : renderAdHocNotice()}<div class="governance"><strong>Governance:</strong> human HSSE review is required for persisted queue reports. The model output is a ranking/triage signal, not an autonomous SIF determination.</div></article>`;
  if (reviewAllowed) bindReviewPanel();
}
function renderAdHocNotice() {
  return `<section class="review-panel review-locked"><div class="subhead"><div><h3>Human HSSE review</h3><p class="muted">Persistent review is available after this narrative is linked to an existing queue report.</p></div><span class="decision unreviewed">ANALYSIS ONLY</span></div></section>`;
}
function renderReviewPanel(d, reviews) {
  const latest = reviews[0];
  return `<section class="review-panel"><div class="subhead"><div><h3>Human HSSE review</h3><p class="muted">Save an explicit decision against this report.</p></div>${latest ? decisionPill(latest.decision) : '<span class="decision unreviewed">UNREVIEWED</span>'}</div>${latest ? `<div class="last-review"><strong>Latest review:</strong> ${escapeHtml(latest.reviewer)} · ${escapeHtml(latest.created_at)}${latest.note ? `<br><span>${escapeHtml(latest.note)}</span>` : ""}</div>` : ""}<div class="review-form"><select id="reviewDecision"><option value="CONFIRMED">CONFIRMED — retain for follow-up</option><option value="REJECTED">REJECTED — not supported by review</option><option value="NEEDS_MORE_INFO">NEEDS MORE INFO — material uncertainty remains</option></select><input id="reviewer" maxlength="120" placeholder="Reviewer name" value="${escapeHtml(latest?.reviewer || "")}"><textarea id="reviewNote" maxlength="4000" rows="3" placeholder="Reason / review note (recommended)"></textarea><button class="primary" id="saveReview">Save human review</button></div></section>`;
}
function bindReviewPanel() {
  const btn = $("saveReview");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const reportId = $("reportId").value.trim();
    if (!reportId) {
      toast("A report ID is required to save a review.", "error");
      return;
    }
    const reviewer = $("reviewer").value.trim() || "demo-reviewer";
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      await api("/reviews", {
        method: "POST",
        body: JSON.stringify({
          report_id: reportId,
          decision: $("reviewDecision").value,
          reviewer,
          note: $("reviewNote").value.trim(),
        }),
      });
      toast("Human review saved.", "success");
      const refreshed = await api(`/reports/${encodeURIComponent(reportId)}`);
      state.selectedReport = refreshed;
      state.analysisContext = "existing";
      renderPrediction(refreshed);
      loadQueue();
      loadOverview();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Save human review";
    }
  });
}
async function runAnalysis() {
  const description = $("description").value.trim();
  if (!description) {
    toast("Enter a narrative first.", "error");
    $("description").focus();
    return;
  }
  const btn = $("analyzeBtn");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  $("prediction").classList.add("hidden");
  try {
    const d = await api("/predict", {
      method: "POST",
      body: JSON.stringify({
        report_id: $("reportId").value.trim() || null,
        description,
      }),
    });
    state.selectedReport = d;
    if ($("reportId").value.trim()) state.analysisContext = "existing";
    renderPrediction(d);
    $("prediction").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}
function clearAnalysis() {
  startNewAnalysis();
}

$("queueSearch").addEventListener("input", renderQueue);
$("clearQueueSearch").addEventListener("click", () => {
  $("queueSearch").value = "";
  renderQueue();
  $("queueSearch").focus();
});
$("priorityFilter").addEventListener("change", loadQueue);
$("statusFilter").addEventListener("change", renderQueue);
$("clearQueueFilters").addEventListener("click", () => {
  $("queueSearch").value = "";
  $("priorityFilter").value = "";
  $("statusFilter").value = "";
  loadQueue();
});
$("clearQueueFiltersInline").addEventListener("click", () =>
  $("clearQueueFilters").click(),
);
$("refreshQueue").addEventListener("click", loadQueue);
$("refreshOverview").addEventListener("click", loadOverview);
$("refreshAnalytics")?.addEventListener("click", loadAnalytics);
$("expandAnalytics")?.addEventListener("click", loadAnalytics);
$("startNewAnalysis").addEventListener("click", startNewAnalysis);
$("startNewAnalysisInline").addEventListener("click", startNewAnalysis);
$("clearAnalysis").addEventListener("click", clearAnalysis);
$("clearAnalysisSecondary").addEventListener("click", clearAnalysis);
$("analyzeBtn").addEventListener("click", runAnalysis);
$("description").addEventListener("input", updateDescriptionCount);

document.addEventListener("keydown", (event) => {
  const tag = (event.target?.tagName || "").toLowerCase();
  if (event.key === "/" && !["input", "textarea", "select"].includes(tag)) {
    event.preventDefault();
    switchTab("queue");
    setTimeout(() => $("queueSearch").focus(), 30);
  }
  if (event.key === "Escape" && document.activeElement === $("queueSearch")) {
    $("clearQueueSearch").click();
  }
  if (
    event.key === "Enter" &&
    (event.ctrlKey || event.metaKey) &&
    document.activeElement === $("description")
  ) {
    event.preventDefault();
    runAnalysis();
  }
});

updateDescriptionCount();
updateAnalysisContextUI();
loadHealth();
loadOverview();
