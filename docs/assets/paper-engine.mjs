/** Deterministic synthetic markets and marked-to-market paper trades in CAD. */

export const PAPER_RULES = Object.freeze({
  initialCash: 100_000,
  sessions: 30,
  contractUsd: 100_000,
  maxShortStraddles: 5,
  reservePerStraddle: 12_000,
});

export const SCENARIOS = Object.freeze([
  Object.freeze({
    id: "cad-selloff",
    label: "CAD selloff",
    description: "The Canadian dollar weakens sharply on session 7 as Canadian equities drop.",
    demoSteps: 7,
  }),
  Object.freeze({
    id: "range",
    label: "Quiet range",
    description: "Narrow FX moves and small equity swings give time decay a larger role.",
    demoSteps: 7,
  }),
  Object.freeze({
    id: "whipsaw",
    label: "FX whipsaw",
    description: "A stronger CAD on session 5 reverses into a sharp depreciation on session 11.",
    demoSteps: 11,
  }),
]);

const INITIAL = Object.freeze({
  equity: 54.48,
  spot: 1.40662,
  vol: 0.04483,
  cadRate: 0.0225,
  usdRate: 0.0426,
});
const INITIAL_TENOR = 0.25;
const DAYS_PER_YEAR = 252;
const EPSILON = 1e-7;

// The large returns are authored synthetic shocks. Remaining returns are
// seeded random moves, with a modest negative equity/FX correlation.
const PATH_SPEC = {
  "cad-selloff": {
    seed: 9137, equityScale: 0.006, fxScale: 0.003,
    events: {7: [-0.052, 0.052, 0.038], 8: [-0.017, 0.014, 0.008], 17: [0.026, -0.018, 0.005]},
  },
  range: {
    seed: 8417, equityScale: 0.004, fxScale: 0.0018,
    events: {},
  },
  whipsaw: {
    seed: 4583, equityScale: 0.008, fxScale: 0.004,
    events: {5: [0.021, -0.052, 0.029], 11: [-0.023, 0.103, 0.042], 18: [0.034, -0.046, 0.016]},
  },
};

function makePath({seed, equityScale, fxScale, events}) {
  function uniform() {
    seed = (Math.imul(1_664_525, seed) + 1_013_904_223) >>> 0;
    return (seed + 0.5) / 2 ** 32;
  }
  const path = [INITIAL];
  for (let day = 1; day <= PAPER_RULES.sessions; day += 1) {
    const previous = path[day - 1];
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = 2 * Math.PI * uniform();
    const z1 = radius * Math.cos(angle);
    const z2 = radius * Math.sin(angle);
    const [equityReturn, fxReturn, volJump] = events[day] ?? [
      equityScale * z1, fxScale * (-0.35 * z1 + 0.937 * z2), 0,
    ];
    path.push({
      equity: previous.equity * Math.exp(equityReturn),
      spot: previous.spot * Math.exp(fxReturn),
      vol: INITIAL.vol + 0.88 * (previous.vol - INITIAL.vol) + volJump,
      cadRate: previous.cadRate,
      usdRate: previous.usdRate,
    });
  }
  return path;
}

const MARKET_PATHS = Object.fromEntries(
  Object.entries(PATH_SPEC).map(([id, spec]) => [id, makePath(spec)]),
);

function density(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function normalCdf(x) {
  // Abramowitz–Stegun 26.2.17, maximum absolute error below 7.5e-8.
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const polynomial = t * (0.319381530 + t * (-0.356563782 +
    t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const lowerTail = density(x) * polynomial;
  return x < 0 ? lowerTail : 1 - lowerTail;
}

function optionTerms({spot, vol, cadRate, usdRate}, tenor) {
  const strike = INITIAL.spot;
  const volTime = vol * Math.sqrt(tenor);
  const d1 = (Math.log(spot / strike) + (cadRate - usdRate + vol ** 2 / 2) * tenor) / volTime;
  const d2 = d1 - volTime;
  const foreignDiscount = Math.exp(-usdRate * tenor);
  const domesticDiscount = Math.exp(-cadRate * tenor);
  return {strike, volTime, d1, d2, foreignDiscount, domesticDiscount};
}

function remainingYears(day) {
  return INITIAL_TENOR - day / DAYS_PER_YEAR;
}

function optionMark(market, day) {
  const {strike, d1, d2, foreignDiscount, domesticDiscount} =
    optionTerms(market, remainingYears(day));
  const perUsd = market.spot * foreignDiscount * (2 * normalCdf(d1) - 1) -
    strike * domesticDiscount * (2 * normalCdf(d2) - 1);
  return PAPER_RULES.contractUsd * Math.max(0, perUsd);
}

function portfolioValue(state, market, straddlePrice = optionMark(market, state.day)) {
  return state.cash + state.xiuShares * market.equity - state.shortStraddles * straddlePrice;
}

function cashReserve(shortStraddles) {
  return shortStraddles * PAPER_RULES.reservePerStraddle;
}

function marketFor(state) {
  return MARKET_PATHS[state.scenario][state.day];
}

function historyPoint(state) {
  const market = marketFor(state);
  return {day: state.day, nav: portfolioValue(state, market), spot: market.spot, equity: market.equity};
}

const STRESS = [
  {id: "fx2", label: "USD/CAD +2%", fxMove: 0.02, xiuMove: 0},
  {id: "fx5", label: "USD/CAD +5%", fxMove: 0.05, xiuMove: 0},
  {id: "xiu5", label: "XIU −5%", fxMove: 0, xiuMove: -0.05},
  {id: "combined", label: "USD/CAD +5% / XIU −5%", fxMove: 0.05, xiuMove: -0.05},
];

function stressValues(state, market, nav) {
  return STRESS.map(({id, label, fxMove, xiuMove}) => ({
    id, label,
    pnl: portfolioValue(state, {
      ...market, spot: market.spot * (1 + fxMove), equity: market.equity * (1 + xiuMove),
    }) - nav,
  }));
}

export function createGame(id = SCENARIOS[0].id) {
  if (!Object.hasOwn(MARKET_PATHS, id)) throw new RangeError("Choose one of the listed market paths.");
  const state = {
    scenario: id, day: 0, cash: PAPER_RULES.initialCash, xiuShares: 0, shortStraddles: 0,
    history: [], trades: [],
  };
  return {...state, history: [historyPoint(state)]};
}

export function snapshot(state) {
  const market = marketFor(state);
  const tenor = remainingYears(state.day);
  const straddle = optionMark(market, state.day);
  const nav = portfolioValue(state, market, straddle);
  const previousNav = state.history.at(-2)?.nav;
  const {d1, foreignDiscount, volTime} = optionTerms(market, tenor);
  const signedStraddles = state.shortStraddles === 0 ? 0 : -state.shortStraddles;
  const gamma = signedStraddles * PAPER_RULES.contractUsd *
    2 * foreignDiscount * density(d1) / (market.spot * volTime);
  const reserve = cashReserve(state.shortStraddles);
  return {
    day: state.day,
    maxDay: PAPER_RULES.sessions,
    spot: market.spot,
    equity: market.equity,
    vol: market.vol,
    remainingYears: tenor,
    remainingDays: INITIAL_TENOR * DAYS_PER_YEAR - state.day,
    positions: {xiu: state.xiuShares, straddle: signedStraddles},
    cash: state.cash,
    reserve,
    availableCash: state.cash - reserve,
    nav,
    pnl: nav - PAPER_RULES.initialCash,
    dailyReturn: previousNav ? nav / previousNav - 1 : 0,
    totalReturn: nav / PAPER_RULES.initialCash - 1,
    stress: stressValues(state, market, nav),
    gamma,
    prices: {xiu: market.equity, straddle},
    strike: INITIAL.spot,
    canAdvance: state.day < PAPER_RULES.sessions,
  };
}

export function placeOrder(state, {instrument, side, quantity}) {
  if (instrument !== "xiu" && instrument !== "straddle") throw new RangeError("Choose an instrument.");
  if (side !== "buy" && side !== "sell") throw new RangeError("Choose buy or sell.");
  if (!Number.isSafeInteger(quantity) || quantity < 1) throw new RangeError("Enter a positive whole quantity.");

  const isEquity = instrument === "xiu";
  const buy = side === "buy";
  const signedQuantity = buy ? quantity : -quantity;
  const xiuShares = state.xiuShares + (isEquity ? signedQuantity : 0);
  const shortStraddles = state.shortStraddles - (isEquity ? 0 : signedQuantity);
  if (xiuShares < 0) throw new RangeError("You can sell only the XIU shares you hold.");
  if (shortStraddles < 0) throw new RangeError("Buybacks are limited to your open short straddles.");
  if (shortStraddles > PAPER_RULES.maxShortStraddles) {
    throw new RangeError("At most five short straddles can be open.");
  }

  const market = marketFor(state);
  const price = isEquity ? market.equity : optionMark(market, state.day);
  const cashFlow = -signedQuantity * price;
  const cash = state.cash + cashFlow;
  if (cash < -EPSILON) throw new RangeError("This order needs more CAD cash.");
  // Selling stock and buying back the option can reduce risk even if the
  // remaining reserve would be underfunded. Cash must always cover buybacks.
  const increasesExposure = isEquity ? buy : !buy;
  if (increasesExposure && cash < cashReserve(shortStraddles) - EPSILON) {
    throw new RangeError("This order would use cash reserved for open short straddles.");
  }

  return {
    ...state, cash: Math.max(0, cash), xiuShares, shortStraddles,
    trades: [...state.trades, {day: state.day, instrument, side, quantity, price, cashFlow}],
  };
}

export function advance(state) {
  if (state.day >= PAPER_RULES.sessions) {
    throw new RangeError("This market path is complete. Reset to start again.");
  }
  const next = {...state, day: state.day + 1};
  return {...next, history: [...state.history, historyPoint(next)]};
}
