import assert from "node:assert/strict";
import test from "node:test";

import {
  PAPER_RULES, SCENARIOS, advance, createGame, placeOrder, snapshot,
} from "../docs/assets/paper-engine.mjs";

function closeTo(value, expected, tolerance = 1e-8) {
  assert.ok(Math.abs(value - expected) < tolerance, `${value} differs from ${expected}`);
}

function order(state, instrument, side, quantity) {
  return placeOrder(state, {instrument, side, quantity});
}

function demoBook(scenario = SCENARIOS[0].id) {
  return order(order(createGame(scenario), "xiu", "buy", 400), "straddle", "sell", 1);
}

test("marks the synthetic option near the Python Garman–Kohlhagen reference", () => {
  const initial = snapshot(createGame());
  assert.equal(initial.spot, initial.strike);
  assert.equal(initial.remainingDays, 63);
  assert.equal(initial.nav, PAPER_RULES.initialCash);
  closeTo(initial.prices.straddle, 2557.68, 0.03);
});

test("fair-mark trades conserve NAV and leave earlier state untouched", () => {
  const initial = createGame();
  const equity = order(initial, "xiu", "buy", 400);
  const book = order(equity, "straddle", "sell", 1);
  closeTo(snapshot(book).nav, PAPER_RULES.initialCash);
  assert.deepEqual(snapshot(book).positions, {xiu: 400, straddle: -1});
  assert.equal(initial.trades.length, 0);
  assert.equal(equity.trades.length, 1);
  assert.equal(book.trades.length, 2);
  closeTo(book.cash, initial.cash - 400 * snapshot(book).equity + snapshot(book).prices.straddle);
  const flat = order(order(book, "straddle", "buy", 1), "xiu", "sell", 400);
  closeTo(snapshot(flat).nav, PAPER_RULES.initialCash);
  closeTo(flat.cash, PAPER_RULES.initialCash);
});

test("rejects invalid orders and protects the cash reserve", () => {
  const initial = createGame();
  assert.throws(() => order(initial, "xiu", "sell", 1), /shares you hold/);
  assert.throws(() => order(initial, "straddle", "buy", 1), /open short/);
  assert.throws(() => order(initial, "straddle", "sell", 6), /At most five/);
  assert.throws(() => order(initial, "xiu", "buy", 1.25), /positive whole/);
  assert.throws(() => order(initial, "xiu", "buy", 10_000), /more CAD cash/);
  const invested = order(initial, "xiu", "buy", 1800);
  assert.throws(() => order(invested, "straddle", "sell", 1), /cash reserved/);
  const short = order(initial, "straddle", "sell", 1);
  assert.equal(snapshot(short).reserve, PAPER_RULES.reservePerStraddle);
  assert.throws(() => order(short, "xiu", "buy", 1800), /cash reserved/);
});

test("marks repeatable session P&L and the default shock", () => {
  let first = demoBook();
  let repeat = demoBook();
  for (let i = 0; i < 6; i += 1) {
    first = advance(first);
    repeat = advance(repeat);
  }
  const preShock = snapshot(first);
  first = advance(first);
  repeat = advance(repeat);
  const postShock = snapshot(first);
  assert.deepEqual(first, repeat);
  assert.equal(first.history.length, 8);
  assert.equal(first.history.at(-1).day, 7);
  assert.ok(postShock.spot > preShock.spot * 1.05);
  assert.ok(postShock.equity < preShock.equity * 0.96);
  assert.ok(postShock.vol > preShock.vol);
  assert.equal(postShock.remainingDays, preShock.remainingDays - 1);
  closeTo(postShock.dailyReturn, postShock.nav / preShock.nav - 1);
  closeTo(postShock.pnl, postShock.nav - PAPER_RULES.initialCash);
  closeTo(postShock.nav, 94092.00, 0.01);
});

test("scenario marks isolate exposures and show short option gamma", () => {
  const flat = snapshot(createGame());
  assert.ok(flat.stress.every(({pnl}) => pnl === 0));
  const book = snapshot(demoBook());
  const stress = Object.fromEntries(book.stress.map(({id, pnl}) => [id, pnl]));
  assert.ok(book.gamma < 0);
  assert.ok(stress.fx5 < stress.fx2 && stress.fx2 < 0);
  closeTo(stress.xiu5, -0.05 * book.equity * book.positions.xiu);
  closeTo(stress.combined, stress.fx5 + stress.xiu5);
});

test("paths diverge and stop at the last marked session", () => {
  const range = createGame("range");
  const whipsaw = createGame("whipsaw");
  assert.notEqual(snapshot(advance(range)).spot, snapshot(advance(whipsaw)).spot);
  assert.throws(() => createGame("unknown"), /listed market paths/);
  let state = range;
  for (let i = 0; i < PAPER_RULES.sessions; i += 1) state = advance(state);
  assert.equal(state.history.length, PAPER_RULES.sessions + 1);
  assert.equal(snapshot(state).canAdvance, false);
  assert.throws(() => advance(state), /market path is complete/);
});
