#!/usr/bin/env python3
"""
Dynamic E-Pricing Engine
========================
Demand forecasting + profit/satisfaction-aware price optimisation + feedback loop.

Pipeline
--------
1. DATA        historical sales, competitor price, inventory, customer behaviour
               (views, cart adds, satisfaction), seasonality (day-of-week, annual
               cycle, holiday peaks)
2. FORECAST    Gradient-boosted Poisson demand model (rate per page-view), with
               monotonic constraints (higher price -> lower demand)
3. OPTIMISE    for each product, search a price grid that maximises
                   profit  -  satisfaction penalty  -  price-volatility penalty
               subject to: minimum margin, competitor band, max daily change,
               and an inventory cap (stock / days-to-restock)
4. FEEDBACK    after every day: actual sales -> (a) fast multiplicative bias
               correction, (b) periodic full retraining on the growing history
5. EVALUATE    MAE / RMSE of the demand forecast, revenue & profit uplift vs
               fixed pricing, decision latency and days-to-adapt after a
               competitor price shock

Run:      python dynamic_pricing.py
          python dynamic_pricing.py --seeds 5 --plot

Plug in real data: replace `MarketData`/`Store` with your own feed. The model only
needs a DataFrame with the columns listed in `HISTORY_COLUMNS` (see `make_features`).

Requires: numpy, pandas, scikit-learn  (matplotlib optional, for --plot)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------------
# 0. Catalogue & configuration
# ----------------------------------------------------------------------------
PRODUCTS = [
    dict(name="Wireless Earbuds", cost=22.0, base_price=60.0, beta=1.5, views=420, conv=0.050, amp=0.30, peak=335),
    dict(name="Running Shoes", cost=35.0, base_price=95.0, beta=1.4, views=300, conv=0.045, amp=0.25, peak=130),
    dict(name="Winter Jacket", cost=48.0, base_price=130.0, beta=1.3, views=250, conv=0.040, amp=0.55, peak=355),
]
P = len(PRODUCTS)
COST = np.array([p["cost"] for p in PRODUCTS])
BASE = np.array([p["base_price"] for p in PRODUCTS])
BETA = np.array([p["beta"] for p in PRODUCTS])
CONV = np.array([p["conv"] for p in PRODUCTS])
RESTOCK_CYCLE = 14  # days between replenishments
HOLIDAY_DOY = np.array(list(range(325, 334)) + list(range(355, 360)))

HISTORY_COLUMNS = ["date", "product_id", "price", "comp_price", "views", "cart_adds",
                   "inventory", "satisfaction", "lag1", "roll7", "units"]


@dataclass
class Config:
    min_margin: float = 0.15        # price >= cost * (1 + min_margin)
    max_daily_change: float = 0.10  # max +/-10% price move per day
    grid_points: int = 41
    floor_vs_comp: float = 0.80     # stay within [0.8x, 1.3x] of competitor
    ceil_vs_comp: float = 1.30
    sat_weight: float = 0.25        # long-term revenue value of customer satisfaction
    change_penalty: float = 0.5     # discourages erratic price swings
    stock_cap_slack: float = 1.1
    retrain_every: int = 14         # feedback loop: full refit cadence (days)
    bias_alpha: float = 0.15        # feedback loop: speed of daily bias correction


def fairness(r):
    """1 = customers see the price as fair (<= 5% above competitor), 0 = very unfair."""
    return 1.0 - np.clip((r - 1.05) / 0.25, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 1. Synthetic market (ground truth) - swap for real data in production
# ----------------------------------------------------------------------------
class MarketData:
    def __init__(self, n_days, seed=0, shock_day=None, shock_size=-0.15, start="2023-01-01"):
        rng = np.random.default_rng(seed)
        self.n, self.shock_day = n_days, shock_day
        self.dates = pd.date_range(start, periods=n_days, freq="D")
        doy_v = self.dates.dayofyear.to_numpy()
        doy, dow = doy_v[:, None], self.dates.dayofweek.to_numpy()[:, None]
        amp = np.array([p["amp"] for p in PRODUCTS])[None, :]
        peak = np.array([p["peak"] for p in PRODUCTS])[None, :]
        holiday = np.isin(doy_v, HOLIDAY_DOY)[:, None]

        # seasonality index (annual cycle + holiday spikes)
        self.season = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25) + 0.35 * holiday
        views_mean = np.array([p["views"] for p in PRODUCTS])[None, :] * self.season * (1 + 0.10 * (dow >= 5))
        self.views = np.round(views_mean * rng.lognormal(0, 0.10, (n_days, P)))

        # latent demand shock; customer-behaviour signal (cart adds) leaks part of it
        self.eps = rng.lognormal(0, 0.10, (n_days, P))
        self.u = rng.uniform(size=(n_days, P))
        self.cart_rate = np.clip(0.08 * self.eps ** 0.7 * rng.lognormal(0, 0.08, (n_days, P)), 0.02, 0.25)

        # competitor price: mean-reverting walk, holiday discounts, optional price war
        comp, x = np.empty((n_days, P)), np.zeros(P)
        for t in range(n_days):
            x = 0.95 * x + rng.normal(0, 0.012, P)
            comp[t] = BASE * 0.98 * np.exp(x)
        comp *= 1 - 0.06 * holiday
        if shock_day is not None:
            comp[shock_day:] *= 1 + shock_size
        self.comp = comp

        # replenishment planned from the seasonal forecast
        base_daily = views_mean * CONV[None, :]
        n_cycles = n_days // RESTOCK_CYCLE + 1
        self.restock = np.zeros((n_cycles, P))
        for c in range(n_cycles):
            sl = slice(c * RESTOCK_CYCLE, min((c + 1) * RESTOCK_CYCLE, n_days))
            self.restock[c] = np.round(base_daily[sl].sum(0) * 1.2 * rng.lognormal(0, 0.08, P))

    def mean_demand(self, t, prices, sat):
        """True expected units. prices: (P,) or (P, G)."""
        prices = np.asarray(prices, float)
        sh = (slice(None), None) if prices.ndim == 2 else slice(None)
        beta = BETA / (1 + 0.5 * (self.season[t] - 1))      # peak season -> less price sensitive
        r = prices / self.comp[t][sh]
        rate = CONV[sh] * np.exp(-beta[sh] * (r - 1)) * (0.8 + 0.4 * sat[sh])
        return self.views[t][sh] * rate


class RecordBook:
    """Accumulates every (context, price, outcome) row -> training data."""
    KEYS = ["product_id", "date", "price", "comp_price", "views", "cart_adds",
            "inventory", "satisfaction", "lag1", "roll7", "units"]

    def __init__(self, base=None):
        self.base = base
        self.cols = {k: [] for k in self.KEYS}

    def add(self, ctx, prices, units):
        c = self.cols
        c["product_id"].append(np.arange(P))
        c["date"].append(np.full(P, ctx["date"].to_datetime64()))
        c["price"].append(np.asarray(prices, float))
        c["comp_price"].append(ctx["comp"])
        for k in ("views", "cart_adds", "inventory", "satisfaction", "lag1", "roll7"):
            c[k].append(ctx[k])
        c["units"].append(np.asarray(units, float))

    def frame(self):
        new = None
        if self.cols["price"]:
            new = pd.DataFrame({k: np.concatenate(v) for k, v in self.cols.items()})
            new["date"] = new["date"].astype("datetime64[ns]")
        if self.base is None:
            return new
        return self.base if new is None else pd.concat([self.base, new], ignore_index=True)


class Store:
    """Mutable shop state: inventory, satisfaction, last price, sales history."""

    def __init__(self, env):
        self.env = env
        self.inv = np.zeros(P)
        self.sat = np.full(P, 0.8)
        self.last_price = BASE.copy()
        self.hist_units = [[p["views"] * p["conv"]] for p in PRODUCTS]

    def clone(self):
        import copy
        s = copy.copy(self)
        s.inv, s.sat, s.last_price = self.inv.copy(), self.sat.copy(), self.last_price.copy()
        s.hist_units = [list(h) for h in self.hist_units]
        return s

    def context(self, t):
        e = self.env
        if t % RESTOCK_CYCLE == 0:
            self.inv = np.maximum(self.inv, e.restock[t // RESTOCK_CYCLE])
        return dict(
            t=t, date=e.dates[t], comp=e.comp[t], views=e.views[t],
            cart_adds=e.views[t] * e.cart_rate[t], inventory=self.inv.copy(),
            satisfaction=self.sat.copy(),
            lag1=np.array([h[-1] for h in self.hist_units]),
            roll7=np.array([np.mean(h[-7:]) for h in self.hist_units]),
            days_to_restock=RESTOCK_CYCLE - t % RESTOCK_CYCLE,
            last_price=self.last_price.copy())

    def sell(self, ctx, prices):
        e, t = self.env, ctx["t"]
        mu = e.mean_demand(t, prices, self.sat) * e.eps[t]
        raw = np.floor(mu) + (e.u[t] < mu - np.floor(mu))     # stochastic rounding
        units = np.minimum(raw, self.inv)                      # censored by stock
        self.inv = self.inv - units
        jump = np.minimum(np.abs(prices / self.last_price - 1), 0.2)
        target = fairness(prices / e.comp[t]) * (1 - jump)
        self.sat = np.clip(0.85 * self.sat + 0.15 * target, 0, 1)
        self.last_price = np.asarray(prices, float).copy()
        for p in range(P):
            self.hist_units[p].append(units[p])
        return units


def simulate_history(env, n_days, seed):
    """Past operations under a noisy 'human' pricing policy (gives price variation to learn from)."""
    rng = np.random.default_rng(seed + 1000)
    store, book = Store(env), RecordBook()
    m, sale_left = np.zeros(P), np.zeros(P)
    for t in range(n_days):
        ctx = store.context(t)
        m = 0.9 * m + rng.normal(0, 0.04, P)
        start_sale = (rng.uniform(size=P) < 0.04) & (sale_left == 0)
        sale_left = np.where(start_sale, 4, np.maximum(sale_left - 1, 0))
        anchor = 0.5 * BASE + 0.5 * ctx["comp"] * 1.02
        prices = np.maximum(anchor * (1 + np.clip(m, -0.2, 0.2)) * np.where(sale_left > 0, 0.88, 1.0),
                            COST * 1.1)
        units = store.sell(ctx, prices)
        book.add(ctx, prices, units)
    return book.frame(), store


# ----------------------------------------------------------------------------
# 2. Demand forecasting
# ----------------------------------------------------------------------------
FEATURES = ["product_id", "price_to_comp", "price_to_base", "comp_to_base", "dow", "sin_doy",
            "cos_doy", "holiday", "cart_rate", "cover", "satisfaction", "lag1", "roll7"]
MONOTONE = [-1 if f in ("price_to_comp", "price_to_base") else (1 if f == "satisfaction" else 0)
            for f in FEATURES]


def make_features(df):
    d = pd.DatetimeIndex(df["date"])
    pid = df["product_id"].to_numpy().astype(int)
    doy = d.dayofyear.to_numpy()
    price, comp = df["price"].to_numpy(), df["comp_price"].to_numpy()
    return pd.DataFrame({
        "product_id": pid,
        "price_to_comp": price / comp,
        "price_to_base": price / BASE[pid],
        "comp_to_base": comp / BASE[pid],
        "dow": d.dayofweek.to_numpy(),
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "holiday": np.isin(doy, HOLIDAY_DOY).astype(float),
        "cart_rate": df["cart_adds"].to_numpy() / np.maximum(df["views"].to_numpy(), 1),
        "cover": df["inventory"].to_numpy() / (df["roll7"].to_numpy() + 1),
        "satisfaction": df["satisfaction"].to_numpy(),
        "lag1": df["lag1"].to_numpy(),
        "roll7": df["roll7"].to_numpy(),
    })[FEATURES]


def fit_demand_model(df, seed=0):
    """Poisson rate model: units / views, weighted by views (views = exposure)."""
    w = np.maximum(df["views"].to_numpy(), 1)
    y = df["units"].to_numpy() / w
    model = HistGradientBoostingRegressor(
        loss="poisson", monotonic_cst=MONOTONE, learning_rate=0.05, max_iter=300,
        max_leaf_nodes=12, min_samples_leaf=25, l2_regularization=1.0, random_state=seed)
    return model.fit(make_features(df), y, sample_weight=w)


def predict_units(model, df, bias=None):
    out = model.predict(make_features(df)) * df["views"].to_numpy()
    if bias is not None:
        out = out * bias[df["product_id"].to_numpy().astype(int)]
    return out


def evaluate_forecast(hist_df, seed=0):
    """Chronological 80/20 split: MAE & RMSE vs naive baselines."""
    cut = hist_df["date"].sort_values().iloc[int(0.8 * len(hist_df))]
    tr, te = hist_df[hist_df["date"] < cut], hist_df[hist_df["date"] >= cut]
    model = fit_demand_model(tr, seed)
    inv = te["inventory"].to_numpy()
    preds = {
        "GBM demand model": np.minimum(predict_units(model, te), inv),
        "7-day moving avg": np.minimum(te["roll7"].to_numpy(), inv),
        "Yesterday's sales": np.minimum(te["lag1"].to_numpy(), inv),
    }
    y = te["units"].to_numpy()
    rows = [dict(model=k, MAE=mean_absolute_error(y, v),
                 RMSE=np.sqrt(mean_squared_error(y, v))) for k, v in preds.items()]
    return pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------
# 3. Price optimisation
# ----------------------------------------------------------------------------
def optimise(grid, units, ctx, cfg, smooth=True):
    """Pick the utility-maximising price per product from a (P, G) grid."""
    cap = np.maximum(ctx["inventory"] / ctx["days_to_restock"] * cfg.stock_cap_slack, 0.0)[:, None]
    sold = np.minimum(units, cap)
    revenue = grid * sold
    profit = (grid - COST[:, None]) * sold
    util = profit - cfg.sat_weight * revenue * (1 - fairness(grid / ctx["comp"][:, None]))
    if smooth:
        util = util - cfg.change_penalty * revenue * np.abs(grid / ctx["last_price"][:, None] - 1)
    util = util - 1e-9 * np.abs(grid - ctx["last_price"][:, None])   # tie-breaker (e.g. empty stock)
    j = util.argmax(axis=1)
    return grid[np.arange(grid.shape[0]), j], j


def oracle_prices(env, ctx, cfg):
    """Best price knowing the TRUE demand curve (used only to measure adaptation speed)."""
    lo = np.maximum(ctx["comp"] * cfg.floor_vs_comp, COST * (1 + cfg.min_margin))
    hi = np.maximum(ctx["comp"] * cfg.ceil_vs_comp, lo)
    grid = np.linspace(lo, hi, 121, axis=1)
    units = env.mean_demand(ctx["t"], grid, ctx["satisfaction"])
    return optimise(grid, units, ctx, cfg, smooth=False)[0]


# ----------------------------------------------------------------------------
# 4. Policies (fixed baseline, rule baseline, dynamic with / without feedback)
# ----------------------------------------------------------------------------
class FixedPolicy:
    name, track_oracle = "Fixed price", False

    def price(self, ctx):
        return BASE.copy(), np.full(P, np.nan)

    def observe(self, *a):
        pass


class CompetitorMatchPolicy:
    name, track_oracle = "Match competitor (rule)", False

    def __init__(self, cfg):
        self.cfg = cfg

    def price(self, ctx):
        return np.maximum(ctx["comp"], COST * (1 + self.cfg.min_margin)).round(2), np.full(P, np.nan)

    def observe(self, *a):
        pass


class DynamicPricer:
    track_oracle = True

    def __init__(self, cfg, hist_df, feedback=True, seed=0):
        self.cfg, self.feedback, self.seed = cfg, feedback, seed
        self.name = "Dynamic + feedback loop" if feedback else "Dynamic, static model"
        self.book = RecordBook(base=hist_df)
        self.bias = np.ones(P)
        self.since_fit = 0
        self.model = fit_demand_model(hist_df, seed)

    def price(self, ctx):
        cfg, G = self.cfg, self.cfg.grid_points
        lo = np.maximum.reduce([ctx["last_price"] * (1 - cfg.max_daily_change),
                                ctx["comp"] * cfg.floor_vs_comp, COST * (1 + cfg.min_margin)])
        hi = np.minimum(ctx["last_price"] * (1 + cfg.max_daily_change), ctx["comp"] * cfg.ceil_vs_comp)
        hi = np.maximum(hi, lo)
        grid = np.linspace(lo, hi, G, axis=1)
        rep = lambda a: np.repeat(a, G)
        df = pd.DataFrame({
            "product_id": rep(np.arange(P)), "date": ctx["date"], "price": grid.ravel(),
            "comp_price": rep(ctx["comp"]), "views": rep(ctx["views"]),
            "cart_adds": rep(ctx["cart_adds"]), "inventory": rep(ctx["inventory"]),
            "satisfaction": rep(ctx["satisfaction"]), "lag1": rep(ctx["lag1"]), "roll7": rep(ctx["roll7"])})
        units = predict_units(self.model, df, self.bias).reshape(P, G)
        prices, j = optimise(grid, units, ctx, cfg)
        return prices.round(2), units[np.arange(P), j]

    def observe(self, ctx, prices, units_obs, pred_raw):
        self.book.add(ctx, prices, units_obs)
        if not self.feedback:
            return
        ok = units_obs < ctx["inventory"]                      # ignore stock-out (censored) days
        ratio = (units_obs + 1) / (pred_raw + 1)
        self.bias = np.where(ok, np.clip(self.bias * ratio ** self.cfg.bias_alpha, 0.6, 1.6), self.bias)
        self.since_fit += 1
        if self.since_fit >= self.cfg.retrain_every:           # periodic retrain on all data so far
            self.model = fit_demand_model(self.book.frame(), self.seed)
            self.bias[:] = 1.0
            self.since_fit = 0


# ----------------------------------------------------------------------------
# 5. Online simulation & evaluation
# ----------------------------------------------------------------------------
def run_policy(policy, env, store0, start, end, cfg):
    store = store0.clone()
    n = end - start
    rec = {k: np.zeros((n, P)) for k in ("price", "units", "pred", "oracle", "sat", "inv_start", "inv_end")}
    latency = []
    for i, t in enumerate(range(start, end)):
        ctx = store.context(t)
        t0 = time.perf_counter()
        prices, pred_raw = policy.price(ctx)
        latency.append((time.perf_counter() - t0) * 1000)
        if policy.track_oracle:
            rec["oracle"][i] = oracle_prices(env, ctx, cfg)
        units = store.sell(ctx, prices)
        policy.observe(ctx, prices, units, pred_raw)
        rec["price"][i], rec["units"][i] = prices, units
        rec["pred"][i] = np.minimum(pred_raw, ctx["inventory"])
        rec["sat"][i], rec["inv_start"][i], rec["inv_end"][i] = store.sat, ctx["inventory"], store.inv
    rec["latency_ms"] = np.array(latency)
    return rec


def adaptation_lag(rec, shock_idx, tol=0.08, hold=3):
    """Days after the competitor shock until our price is within `tol` of the oracle for `hold` days."""
    err = np.abs(rec["price"] - rec["oracle"]) / rec["oracle"]
    ok = err < tol
    lags = []
    for p in range(P):
        lag = np.nan
        for i in range(shock_idx, ok.shape[0] - hold + 1):
            if ok[i:i + hold, p].all():
                lag = i - shock_idx
                break
        lags.append(lag)
    return np.array(lags)


def run_experiment(seed, hist_days=730, online_days=180, shock_offset=90, cfg=None, verbose=True):
    cfg = cfg or Config()
    n = hist_days + online_days
    env = MarketData(n, seed=seed, shock_day=hist_days + shock_offset, shock_size=-0.15)
    hist_df, store0 = simulate_history(env, hist_days, seed)

    fc = evaluate_forecast(hist_df, seed)
    policies = [FixedPolicy(), CompetitorMatchPolicy(cfg),
                DynamicPricer(cfg, hist_df, feedback=False, seed=seed),
                DynamicPricer(cfg, hist_df, feedback=True, seed=seed)]
    runs = {p.name: run_policy(p, env, store0, hist_days, n, cfg) for p in policies}

    rows = []
    for name, r in runs.items():
        rev = (r["price"] * r["units"]).sum()
        prof = ((r["price"] - COST) * r["units"]).sum()
        row = dict(policy=name, revenue=rev, profit=prof, units=r["units"].sum(),
                   avg_satisfaction=r["sat"].mean(), stockout_rate=(r["inv_end"] <= 0).mean())
        if not np.isnan(r["pred"]).any():
            err = r["pred"] - r["units"]
            row.update(online_MAE=np.abs(err).mean(), online_RMSE=np.sqrt((err ** 2).mean()))
        rows.append(row)
    summ = pd.DataFrame(rows).set_index("policy")
    fixed = summ.loc["Fixed price"]
    summ["revenue_uplift_%"] = 100 * (summ["revenue"] / fixed["revenue"] - 1)
    summ["profit_uplift_%"] = 100 * (summ["profit"] / fixed["profit"] - 1)

    resp = {}
    for name in ("Dynamic, static model", "Dynamic + feedback loop"):
        r = runs[name]
        lag = adaptation_lag(r, shock_offset)
        resp[name] = dict(latency_mean_ms=r["latency_ms"].mean(),
                          latency_p95_ms=np.percentile(r["latency_ms"], 95),
                          days_to_adapt_after_shock=np.nanmean(lag) if not np.isnan(lag).all() else np.nan)
    resp = pd.DataFrame(resp).T

    if verbose:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
        print(f"\n=== seed {seed} | history {hist_days}d | online {online_days}d | "
              f"competitor price war (-15%) on online day {shock_offset} ===")
        print("\n[1] Demand forecast on hold-out history (units/product/day)")
        print(fc)
        print("\n[2] Online results (same market, same random demand shocks for every policy)")
        print(summ[["revenue", "profit", "revenue_uplift_%", "profit_uplift_%", "avg_satisfaction",
                    "stockout_rate", "online_MAE", "online_RMSE"]])
        print("\n[3] Response time to market changes")
        print(resp)
    return dict(fc=fc, summary=summ, resp=resp, runs=runs, env=env, hist_df=hist_df,
                hist_days=hist_days, shock_offset=shock_offset)


def plot_results(res, path="dynamic_pricing_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return
    runs, env, h, so = res["runs"], res["env"], res["hist_days"], res["shock_offset"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, r in runs.items():
        ax[0].plot(np.cumsum(((r["price"] - COST) * r["units"]).sum(1)), label=name)
    ax[0].axvline(so, color="grey", ls="--")
    ax[0].set(title="Cumulative profit", xlabel="online day", ylabel="profit")
    ax[0].legend(fontsize=8)
    d = runs["Dynamic + feedback loop"]
    days = np.arange(d["price"].shape[0])
    ax[1].plot(days, d["price"][:, 2], label="our price")
    ax[1].plot(days, env.comp[h:, 2], label="competitor", alpha=.7)
    ax[1].plot(days, d["oracle"][:, 2], label="oracle (true optimum)", ls=":")
    ax[1].axvline(so, color="grey", ls="--")
    ax[1].set(title=f"{PRODUCTS[2]['name']}: price tracking", xlabel="online day", ylabel="price")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, help="number of random markets to evaluate")
    ap.add_argument("--hist-days", type=int, default=730)
    ap.add_argument("--online-days", type=int, default=180)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    results = [run_experiment(s, a.hist_days, a.online_days, verbose=(s == 0)) for s in range(a.seeds)]
    if a.seeds > 1:
        up = pd.concat([r["summary"][["revenue_uplift_%", "profit_uplift_%"]] for r in results])
        agg = up.groupby(level=0).agg(["mean", "std"])
        print(f"\n=== uplift vs fixed pricing over {a.seeds} random markets ===")
        print(agg.round(2))
    if a.plot:
        plot_results(results[0])


if __name__ == "__main__":
    main()
