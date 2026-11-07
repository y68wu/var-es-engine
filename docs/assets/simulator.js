import {
  PAPER_RULES, SCENARIOS, advance, createGame, placeOrder, snapshot,
} from "./paper-engine.mjs";

const get = (id) => document.getElementById(id);
const chart = get("sim-chart");
const mode = get("sim-mode");
const instrument = get("sim-instrument");
const side = get("sim-side");
const quantity = get("sim-qty");
const chartButtons = [...document.querySelectorAll("[data-sim-chart]")];
const money = new Intl.NumberFormat("en-CA", {
  style: "currency", currency: "CAD", maximumFractionDigits: 2, minimumFractionDigits: 2,
});
const integers = new Intl.NumberFormat("en-CA", { maximumFractionDigits: 0 });
const pct = new Intl.NumberFormat("en-CA", { style: "percent", maximumFractionDigits: 2 });
const actions = {
  buy_xiu: "Buy XIU",
  sell_xiu: "Sell XIU",
  sell_straddle: "Sell straddle to open",
  buy_straddle: "Buy straddle to close",
};
const signed = (value) => (value < 0 ? "−" : value > 0 ? "+" : "") + money.format(Math.abs(value));

let state = createGame("cad-selloff");
let chartMode = "market";
let timer = null;

function announce(text, kind = "info") {
  const status = get("sim-status");
  status.textContent = text;
  status.dataset.kind = kind;
}

function row(table, values) {
  const tr = document.createElement("tr");
  values.forEach((value) => {
    const td = document.createElement("td");
    td.textContent = value;
    tr.append(td);
  });
  table.append(tr);
}

function setVal(id, value) {
  get(id).textContent = value;
}

function svg(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  chart.append(node);
}

function draw() {
  const nav = chartMode === "nav";
  setVal("sim-chart-caption", nav ? "ACCOUNT NAV / CAD" : "MARKET / SESSION ZERO = 100");
  const first = state.history[0];
  const series = nav
    ? [{ key: "nav", color: "#62f3bd", label: "NAV", values: state.history.map((bar) => bar.nav) }]
    : [
      { key: "xiu", color: "#62f3bd", label: "XIU", values: state.history.map((bar) => 100 * bar.equity / first.equity) },
      { key: "fx", color: "#a79cf5", label: "USD/CAD", values: state.history.map((bar) => 100 * bar.spot / first.spot) },
    ];
  const values = series.flatMap((item) => item.values);
  const baseline = nav ? PAPER_RULES.initialCash : 100;
  const minimum = Math.min(baseline, ...values);
  const maximum = Math.max(baseline, ...values);
  const padding = Math.max((maximum - minimum) * 0.22, nav ? 350 : 0.6);
  const low = minimum - padding;
  const high = maximum + padding;
  const x0 = 91, y0 = 27, width = 638, height = 178;
  const x = (day) => x0 + day / PAPER_RULES.sessions * width;
  const y = (value) => y0 + (high - value) / (high - low) * height;
  const format = (value) => nav ? `$${(value / 1000).toFixed(1)}k` : value.toFixed(1);
  chart.replaceChildren();
  chart.setAttribute("aria-label", nav
    ? "Portfolio net asset value by simulated session, in Canadian dollars"
    : "Simulated XIU and USD/CAD relative to session zero, indexed to 100");
  for (let index = 0; index <= 4; index += 1) {
    const value = low + (high - low) * index / 4;
    svg("line", { x1: x0, x2: x0 + width, y1: y(value), y2: y(value), class: "sim-grid-line" });
    svg("text", { x: x0 - 10, y: y(value) + 4, "text-anchor": "end", class: "sim-axis-text" }, format(value));
  }
  for (let day = 0; day <= PAPER_RULES.sessions; day += 5) {
    svg("line", { x1: x(day), x2: x(day), y1: y0, y2: y0 + height, class: "sim-grid-line" });
    svg("text", { x: x(day), y: y0 + height + 22, "text-anchor": "middle", class: "sim-axis-text" }, day);
  }
  if (state.day < PAPER_RULES.sessions) {
    svg("rect", {
      x: x(state.day), y: y0, width: x(PAPER_RULES.sessions) - x(state.day), height,
      fill: "#0a1520", opacity: ".43",
    });
  }
  svg("line", {
    x1: x(state.day), x2: x(state.day), y1: y0, y2: y0 + height,
    stroke: "#a0b6c3", "stroke-width": "1", "stroke-dasharray": "4 5", opacity: ".55",
  });
  series.forEach(({ values: points, color, label }, index) => {
    svg("polyline", {
      points: points.map((value, i) => `${x(state.history[i].day)},${y(value)}`).join(" "),
      fill: "none", stroke: color, "stroke-width": 3, "stroke-linejoin": "round",
      "stroke-linecap": "round", "vector-effect": "non-scaling-stroke",
    });
    svg("circle", {
      cx: x(state.day), cy: y(points.at(-1)), r: 4.2, fill: color, stroke: "#091420",
      "stroke-width": 1.3,
    });
    const legendX = x0 + index * 121;
    svg("circle", { cx: legendX, cy: 249, r: 3.4, fill: color });
    svg("text", { x: legendX + 10, y: 253, class: "sim-axis-text" }, label);
  });
  svg("text", { x: x0 + width, y: 253, "text-anchor": "end", class: "sim-axis-text" }, "SESSION");
}

function updateTicket(view) {
  const option = instrument.value === "straddle";
  const price = option ? view.prices.straddle : view.equity;
  const qty = Number(quantity.value);
  const direction = side.value === "buy" ? "debit" : "credit";
  const validQuantity = Number.isInteger(qty) && qty >= 1;
  const unit = get("sim-ticket-unit");
  if (unit) unit.textContent = option
    ? `USD ${integers.format(PAPER_RULES.contractUsd)} per contract · ${view.remainingDays} sessions to expiry · ${pct.format(view.vol)} volatility`
    : "XIU units";
  quantity.max = option ? PAPER_RULES.maxShortStraddles : Math.floor(PAPER_RULES.initialCash);
  setVal("sim-preview", validQuantity
    ? `${money.format(price)} per ${option ? "contract" : "unit"} · ${money.format(qty * price)} cash ${direction}`
    : "Enter a positive whole number.");
  get("sim-submit").disabled = !validQuantity;
  get("sim-submit").textContent = actions[`${side.value}_${instrument.value}`];
}

function render() {
  const view = snapshot(state);
  const positionTable = get("sim-positions");
  const tradeTable = get("sim-trades");
  const stress = view.stress.find((item) => item.id === "fx2");
  setVal("sim-step", "REPLAY SESSION");
  setVal("sim-clock", `${String(view.day).padStart(2, "0")} / ${PAPER_RULES.sessions}`);
  setVal("sim-equity", money.format(view.equity));
  setVal("sim-spot", view.spot.toFixed(4));
  setVal("sim-nav", money.format(view.nav));
  setVal("sim-cash", money.format(view.availableCash));
  setVal("sim-pnl", `${signed(view.nav - PAPER_RULES.initialCash)} (${pct.format(view.totalReturn)})`);
  setVal("sim-stress", stress ? signed(stress.pnl) : "—");
  setVal("sim-gamma", view.positions.straddle < 0 ? `Short gamma · ${Math.abs(view.positions.straddle)} short` : "No option exposure");
  get("sim-next").disabled = !view.canAdvance || timer !== null;
  get("sim-run").disabled = !view.canAdvance;
  get("sim-run").textContent = timer === null ? "Play" : "Pause";
  get("sim-run").setAttribute("aria-pressed", String(timer !== null));

  positionTable.replaceChildren();
  row(positionTable, ["XIU", integers.format(view.positions.xiu), money.format(view.equity), money.format(view.positions.xiu * view.equity)]);
  row(positionTable, ["USD/CAD straddle", integers.format(view.positions.straddle),
    money.format(view.prices.straddle), signed(view.positions.straddle * view.prices.straddle)]);
  row(positionTable, ["CAD cash", "—", "—", money.format(view.cash)]);
  const shockTable = get("sim-shocks");
  if (shockTable) {
    shockTable.replaceChildren();
    view.stress.forEach((scenario) => row(shockTable, [scenario.label, signed(scenario.pnl)]));
  }

  tradeTable.replaceChildren();
  if (state.trades.length === 0) {
    row(tradeTable, ["—", "No trades yet", "—", "—", "—"]);
  } else {
    state.trades.slice().reverse().forEach((trade) => {
      row(tradeTable, [String(trade.day).padStart(2, "0"), actions[`${trade.side}_${trade.instrument}`],
        integers.format(trade.quantity), money.format(trade.price), signed(trade.cashFlow)]);
    });
  }
  updateTicket(view);
  draw();
}

function stop() {
  if (timer === null) return;
  clearInterval(timer);
  timer = null;
}

function tick() {
  state = advance(state);
  if (!snapshot(state).canAdvance) {
    stop();
    announce("Replay complete. Review your book or reset to start a new run.");
  }
  render();
}

function reset() {
  stop();
  state = createGame(mode.value);
  const selected = SCENARIOS.find((path) => path.id === mode.value);
  setVal("sim-mode-note", `${selected.description} Changing paths resets the account.`);
  announce("New paper account. Choose an order or run the sample.");
  render();
}

for (const path of SCENARIOS) mode.append(new Option(path.label, path.id));
mode.value = "cad-selloff";
mode.addEventListener("change", reset);
get("sim-reset").addEventListener("click", reset);
get("sim-next").addEventListener("click", tick);
get("sim-run").addEventListener("click", () => {
  if (timer === null) {
    timer = setInterval(tick, 650);
    announce("Replay running. You can pause at any session.");
  } else {
    stop();
    announce("Replay paused.");
  }
  render();
});
get("sim-demo").addEventListener("click", () => {
  reset();
  try {
    const selected = SCENARIOS.find((path) => path.id === mode.value);
    state = placeOrder(state, { instrument: "xiu", side: "buy", quantity: 400 });
    state = placeOrder(state, { instrument: "straddle", side: "sell", quantity: 1 });
    for (let index = 0; index < selected.demoSteps; index += 1) state = advance(state);
    announce(`Sample filled 400 XIU units and sold 1 straddle at session 00. Replay advanced to session ${state.day}. Try buying back the straddle or advancing again.`, "success");
    instrument.value = "straddle";
    side.value = "buy";
    quantity.value = 1;
    chartMode = "nav";
    chartButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.simChart === chartMode)));
  } catch (error) {
    announce(error.message, "error");
  }
  render();
});
get("sim-order").addEventListener("submit", (event) => {
  event.preventDefault();
  const action = `${side.value}_${instrument.value}`;
  try {
    stop();
    state = placeOrder(state, { instrument: instrument.value, side: side.value, quantity: Number(quantity.value) });
    const trade = state.trades.at(-1);
    announce(`Filled at session ${trade.day}: ${actions[action]} × ${trade.quantity} at ${money.format(trade.price)}.`, "success");
  } catch (error) {
    announce(error.message, "error");
  }
  render();
});
instrument.addEventListener("change", () => {
  quantity.value = instrument.value === "straddle" ? 1 : 100;
  updateTicket(snapshot(state));
});
side.addEventListener("change", () => updateTicket(snapshot(state)));
quantity.addEventListener("input", () => updateTicket(snapshot(state)));
chartButtons.forEach((button) => button.addEventListener("click", () => {
  chartMode = button.dataset.simChart;
  chartButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  draw();
}));
reset();
