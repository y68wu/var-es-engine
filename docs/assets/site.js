"use strict";

const METHODS = {
  historical: ["Historical simulation", "Full revaluation of the 500 observed joint shocks."],
  delta_normal: ["Delta-normal", "Linear P&L from the historical factor mean and covariance."],
  delta_gamma: ["Delta-gamma", "Quadratic P&L with a Cornish–Fisher tail adjustment."],
  mc_normal: ["MC Gaussian", "10,000 correlated Gaussian paths with full revaluation."],
  mc_t: ["MC Student-t", "10,000 covariance-matched Student-t paths with six degrees of freedom."],
};
const CURVES = {
  delta_gamma_pnl: { color: "#aaa0fc", dash: "2 5" },
  delta_pnl: { color: "#ff927c", dash: "6 5" },
  full_revaluation: { color: "#62f3bd", dash: "" },
};
const select = (selector) => document.querySelector(selector);
const selectAll = (selector) => [...document.querySelectorAll(selector)];
const formatDate = (date) => new Intl.DateTimeFormat("en-CA", {
  dateStyle: "medium", timeZone: "UTC",
}).format(new Date(`${date}T00:00:00Z`));
const formatInt = (value) => new Intl.NumberFormat("en-CA").format(value);
const formatPercent = (value, decimals = 2) => `${(value * 100).toFixed(decimals)}%`;
const formatP = (value) => value < 0.0001 ? "<0.0001" : value.toFixed(4);

function cad(value, signed = false) {
  const abs = Math.abs(value);
  const prefix = value < 0 ? "−" : signed && value > 0 ? "+" : "";
  if (abs >= 1e6) return `${prefix}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${prefix}$${(abs / 1e3).toFixed(1)}k`;
  return `${prefix}$${Math.round(abs)}`;
}

function setPressed(buttons, selected) {
  buttons.forEach((button) => button.setAttribute("aria-pressed", String(button === selected)));
}

function svgElement(tag, attrs, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawShockChart(svg, curve, hero) {
  const { width, height } = svg.viewBox.baseVal;
  const bounds = { left: 72, right: width - 28, top: 18, bottom: height - 32 };
  const axes = hero ? ["delta_pnl", "full_revaluation"] : Object.keys(CURVES);
  const xTicks = [-8, -4, 0, 4, 8];
  const step = 500000;
  const low = Math.floor(Math.min(...curve.flatMap((point) => axes.map((key) => point[key]))) / step) * step;
  const high = Math.ceil(Math.max(...curve.flatMap((point) => axes.map((key) => point[key]))) / step) * step;
  const x = (value) => bounds.left + (value * 100 + 8) * (bounds.right - bounds.left) / 16;
  const y = (value) => bounds.bottom - (value - low) * (bounds.bottom - bounds.top) / (high - low);
  const draw = (tag, attrs, text) => svg.appendChild(svgElement(tag, attrs, text));

  for (let v = low; v <= high; v += step) {
    const level = y(v);
    draw("line", { x1: bounds.left, x2: bounds.right, y1: level, y2: level, class: v === 0 ? "axis-zero" : "axis-grid" });
    const label = Math.abs(v) >= 1e6 ? `${Math.abs(v / 1e6)}M` : `${Math.abs(v / 1000)}k`;
    draw("text", { x: bounds.left - 9, y: level + 4, "text-anchor": "end", class: "axis-label" },
      v === 0 ? "$0" : `${v < 0 ? "−" : "+"}$${label}`);
  }
  for (const tick of xTicks) {
    const position = x(tick / 100);
    draw("line", { x1: position, x2: position, y1: bounds.top, y2: bounds.bottom, class: tick === 0 ? "axis-zero" : "axis-grid" });
    draw("text", { x: position, y: height - 9, "text-anchor": "middle", class: "axis-label" },
      `${tick > 0 ? "+" : tick < 0 ? "−" : ""}${Math.abs(tick)}%`);
  }

  const points = (key) => curve.map((point) => `${x(point.fx_spot_move).toFixed(2)} ${y(point[key]).toFixed(2)}`).join(" L ");
  if (hero) {
    const shading = "hero-shading";
    const defs = draw("defs", {});
    const gradient = svgElement("linearGradient", { id: shading, x2: "0%", y2: "100%" });
    gradient.append(svgElement("stop", { offset: "0%", "stop-color": "#62f3bd", "stop-opacity": ".20" }),
      svgElement("stop", { offset: "100%", "stop-color": "#62f3bd", "stop-opacity": ".02" }));
    defs.append(gradient);
    draw("path", { d: `M ${points("full_revaluation")} L ${bounds.right} ${bounds.bottom} L ${bounds.left} ${bounds.bottom} Z`, fill: `url(#${shading})` });
  }
  for (const key of axes) {
    const { color, dash } = CURVES[key];
    draw("path", { d: `M ${points(key)}`, fill: "none", stroke: color, "stroke-width": key === "full_revaluation" ? 2.8 : 1.9, "stroke-linecap": "round", "stroke-dasharray": dash });
  }
  const line = draw("line", { y1: bounds.top, y2: bounds.bottom, class: "crosshair" });
  const markers = axes.map((key) => [key, draw("circle", { r: 5, fill: CURVES[key].color, stroke: "#091420", "stroke-width": 1.5 })]);
  return (point) => {
    const cursor = x(point.fx_spot_move);
    line.setAttribute("x1", cursor);
    line.setAttribute("x2", cursor);
    for (const [key, marker] of markers) {
      marker.setAttribute("cx", cursor);
      marker.setAttribute("cy", y(point[key]));
    }
  };
}

function buildLab(allPoints) {
  const curve = allPoints.filter((point) => Math.abs(point.fx_spot_move) <= 0.080000001);
  const keyed = new Map(curve.map((point) => [Math.round(point.fx_spot_move * 1000), point]));
  const heroMarker = drawShockChart(select("#hero-chart"), curve, true);
  heroMarker(keyed.get(20));
  const markLab = drawShockChart(select("#lab-chart"), curve, false);
  const slider = select("#shock-range");
  const presets = selectAll("[data-shock]");

  function update() {
    const value = Number(slider.value);
    const point = keyed.get(Math.round(value * 10));
    if (!point) return;
    select("#shock-percent").textContent = `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(1)}%`;
    select("#exact-value").textContent = cad(point.full_revaluation, true);
    select("#linear-value").textContent = cad(point.delta_pnl, true);
    select("#gamma-value").textContent = cad(point.delta_gamma_pnl, true);
    slider.setAttribute("aria-valuetext", `${value} percent`);
    setPressed(presets, presets.find((button) => Number(button.dataset.shock) === value));
    markLab(point);
  }

  slider.addEventListener("input", update);
  presets.forEach((button) => button.addEventListener("click", () => {
    slider.value = button.dataset.shock;
    update();
  }));
  update();
}

function buildModelComparison(snapshot, diagnostics, market) {
  const byMethod = Object.fromEntries(diagnostics.map((row) => [row.method, row]));
  const measures = selectAll("[data-metric]");
  const rows = snapshot.map((risk) => ({ ...risk, ...byMethod[risk.method] }));
  const container = select("#model-bars");
  const detail = select("#model-detail");
  const rowButtons = rows.map(({ method }) => {
    const button = document.createElement("button");
    button.className = "model-row";
    button.type = "button";
    button.dataset.method = method;
    button.innerHTML = `<span>${METHODS[method][0]}</span><span class="model-bar" aria-hidden="true"><i class="model-fill"></i></span><strong></strong>`;
    container.append(button);
    return button;
  });

  function showMethod(index) {
    const { method, kupiec_p, independence_p, conditional_p, exceptions_last_250, traffic_last_250 } = rows[index];
    const [label, about] = METHODS[method];
    setPressed(rowButtons, rowButtons[index]);
    detail.innerHTML = `<div><h3>${label}</h3><p>${about}</p><p>Last 250 days · ${exceptions_last_250} exceptions · ${traffic_last_250}</p></div>
      <dl><div><dt>Kupiec p</dt><dd></dd></div><div><dt>Independence p</dt><dd></dd></div><div><dt>Conditional p</dt><dd></dd></div></dl>`;
    detail.querySelectorAll("dd").forEach((node, i) => {
      node.textContent = formatP([kupiec_p, independence_p, conditional_p][i]);
    });
  }

  function showMeasure(button) {
    const measure = button.dataset.metric;
    const count = measure === "exceptions";
    setPressed(measures, button);
    const caption = market?.as_of ? formatDate(market.as_of) : "Oct 21, 2026";
    select("#metric-caption").textContent = count ? "750 forecasts · expected 7.5 exceptions" : `Reference snapshot · ${caption}`;
    const max = Math.max(...rows.map((row) => row[measure])) * 1.05;
    rowButtons.forEach((row, i) => {
      const value = rows[i][measure];
      const bar = row.querySelector(".model-bar");
      bar.classList.toggle("with-benchmark", count);
      bar.style.setProperty("--benchmark", `${(7.5 / max) * 100}%`);
      row.querySelector(".model-fill").style.width = `${(value / max) * 100}%`;
      row.querySelector("strong").textContent = count ? `${value} / 750` : cad(value);
    });
  }

  measures.forEach((button) => button.addEventListener("click", () => showMeasure(button)));
  rowButtons.forEach((button, i) => button.addEventListener("click", () => showMethod(i)));
  showMeasure(measures[0]);
  showMethod(rows.findIndex(({ method }) => method === "delta_normal"));
}

function buildScenarios(rows) {
  const tabs = selectAll("[data-scenario-group]");
  const picker = select("#scenario-buttons");
  const positionNames = { equity: "XIU", goc_5y: "GoC 5Y", goc_10y: "GoC 10Y", fx_straddle: "FX straddle", cash: "Cash" };

  function showScenario(row) {
    select("#scenario-label").textContent = row.measure === "worst close" ? "WORST CLOSE REPLAY" : row.measure === "period end" ? "PERIOD-END REPLAY" : "INSTANTANEOUS STRESS";
    select("#scenario-date").textContent = row.scenario_date ? formatDate(row.scenario_date).toUpperCase() : "HYPOTHETICAL";
    select("#scenario-name").textContent = row.scenario;
    const pnl = select("#scenario-total");
    pnl.textContent = cad(row.total, true);
    pnl.classList.toggle("positive", row.total > 0);
    const max = Math.max(...Object.keys(positionNames).map((key) => Math.abs(row[key]))) || 1;
    select("#stress-legs").innerHTML = Object.entries(positionNames).map(([key, label]) => {
      const value = row[key];
      const width = Math.abs(value) * 50 / max;
      const left = value < 0 ? 50 - width : 50;
      return `<div class="stress-leg"><span>${label}</span><span class="leg-bar" aria-hidden="true"><i class="leg-fill${value < 0 ? " negative" : ""}" style="left:${left}%;width:${width}%"></i></span><strong>${cad(value, true)}</strong></div>`;
    }).join("");
  }

  function showGroup(tab) {
    setPressed(tabs, tab);
    const historical = tab.dataset.scenarioGroup === "historical";
    const filtered = rows.filter(({ measure }) => historical ? measure !== "instantaneous" : measure === "instantaneous");
    picker.replaceChildren();
    const buttons = filtered.map((row) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = row.scenario;
      if (historical) {
        const subtitle = document.createElement("small");
        subtitle.textContent = row.measure === "worst close" ? "Worst close" : "Period end";
        button.append(subtitle);
      }
      picker.append(button);
      return button;
    });
    buttons.forEach((button, i) => button.addEventListener("click", () => {
      setPressed(buttons, button);
      showScenario(filtered[i]);
    }));
    const first = historical ? filtered.findIndex(({ measure }) => measure === "worst close") : 0;
    setPressed(buttons, buttons[first]);
    showScenario(filtered[first]);
  }

  tabs.forEach((tab) => tab.addEventListener("click", () => showGroup(tab)));
  showGroup(tabs[0]);
}

function buildContext(context) {
  const { fx_validation: fx, vix_regimes: regimes } = context;
  const panel = select("#context-panel");
  const nextSession = fx.lag_correlations?.find(({ yahoo_offset_sessions }) => yahoo_offset_sessions === 1);
  panel.innerHTML = `<h3>What the extra data revealed</h3>
    <p>Two independent public series put the input conventions and backtest exceptions in context.</p>
    <div class="context-grid">
      <article><span class="micro">BANK OF CANADA / FXUSDCAD</span><h4>Same currency pair, different clock</h4>
        <strong class="fx-stat">${fx.median_abs_spot_gap_bp.toFixed(1)} bp</strong>
        <p>Median absolute relative spot gap across ${formatInt(fx.overlap_days)} matched dates. BoC reports a daily average and Yahoo records a close. Same-date return correlation was ${fx.return_correlation.toFixed(3)}${nextSession ? `; it was ${nextSession.correlation.toFixed(3)} when paired with the next aligned Yahoo label` : ""}. The shift shows sensitivity to quote timing.</p>
        <a class="context-source" href="https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/">BoC daily FX rates ↗</a>
      </article>
      <article><span class="micro">FRED / VIXCLS</span><h4>Exceptions by prior VIX level</h4>
        <label class="micro" for="regime-model">SELECT A MODEL</label>
        <select class="context-select" id="regime-model"></select>
        <div class="regime-chart" aria-live="polite"></div>
        <p>VIX uses the most recent observation before each realization date. These counts describe the sample and were excluded from model calibration.</p>
        <a class="context-source" href="https://fred.stlouisfed.org/series/VIXCLS">FRED series ↗</a>
      </article>
    </div>`;
  const picker = panel.querySelector("select");
  for (const [id, [label]] of Object.entries(METHODS)) picker.add(new Option(label, id));
  picker.value = "delta_normal";

  function update() {
    const chart = panel.querySelector(".regime-chart");
    const matches = regimes.groups.map((group) => ({ ...group, ...group.models.find(({ method }) => method === picker.value) }));
    const max = Math.max(...matches.map(({ exception_rate }) => exception_rate), 0.01) * 1.15;
    chart.innerHTML = matches.map(({ id, days, exceptions, exception_rate }) => `<div class="regime-item">
      <span>${id === "below_20" ? "VIX < 20" : "VIX ≥ 20"}</span>
      <span class="regime-bar" aria-hidden="true"><i class="regime-fill" style="width:${exception_rate / max * 100}%"></i></span>
      <strong>${formatPercent(exception_rate)} <small>(${exceptions}/${days})</small></strong>
    </div>`).join("");
  }

  picker.addEventListener("change", update);
  update();
  panel.hidden = false;
}

async function main() {
  try {
    const response = await fetch("assets/results.json");
    if (!response.ok) throw new Error(`Results returned HTTP ${response.status}`);
    const data = await response.json();
    buildLab(data.curve);
    buildModelComparison(data.snapshot, data.diagnostics, data.market);
    buildScenarios(data.stress);
    if (data.market?.start && data.market?.as_of) {
      select("#study-window").textContent = `${formatDate(data.market.start)} — ${formatDate(data.market.as_of)}`;
    }
    if (data.context) buildContext(data.context);
  } catch (error) {
    select("#data-error").hidden = false;
    console.error("Risk Lab initialization failed", error);
  }
}

main();
