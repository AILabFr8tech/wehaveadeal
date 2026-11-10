"""Exact margin/profit percentages from the cached raw_results.csv, without re-simulating.

The market (sale price, goods cost, each shipper's route mean) depends only on the
seed. A matched shipper's rate = sale - goods - profit, and a matched carrier's
margin = rate - mu(route of its shipper), so each carrier is paired with the shipper
whose (rate - mu) equals its margin. Then margin% = margin / mu, profit% = profit / sale.
"""
import sys, os, csv
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT)); os.chdir(ROOT)  # ROOT = paper/
import numpy as np
from wehaveadeal.market import MarketState
K = ["r", "rv", "is", "opt", "opt2", "ml", "ml2", "rl", "rl2"]
LBL = {"r":"R","rv":"RV","is":"IS","opt":"OPT","opt2":"OPT2","ml":"ML","ml2":"ML2","rl":"RL","rl2":"RL2"}

rows = list(csv.DictReader(open("data/validation/raw_results.csv")))
days = defaultdict(list)
for r in rows: days[r["day_id"]].append(r)

markets = {}
def market(seed, n):
    if (seed, n) not in markets: markets[(seed, n)] = MarketState(seed, n)
    return markets[(seed, n)]

fallback_declined = [0]
out = []  # (config, role, kind, pct)
unrecovered = 0; carriers_matched = 0
def day_key(d):
    c, e, s, i = d.split("|"); return (int(s), int(i))
# Spawning advances the SeedSequence counter, so replay run_round's exact call
# order: per seed, negotiations in index order, negotiation_rng then negotiation_seed_int.
for day_id in sorted(days, key=day_key):
    rs = days[day_id]
    config, excl, seed, idx = day_id.split("|"); seed = int(seed); idx = int(idx)
    n = 400 if config == "balanced" else 12
    m = market(seed, n)
    shipper_kinds = [k for k in K if not (config == "shipper_scarce" and k in excl.split("+"))]
    rng = m.negotiation_rng(idx)
    mu = {}
    for k in shipper_kinds:
        route = m.assign_route(rng); mu[k] = route.refprice_mean
        m.sample_shipper_epsilon(route.refprice_sigma, rng)
    m.negotiation_seed_int(idx)
    sale, goods = m.sale_prices[idx], m.goods_costs[idx]
    cand = {}
    declined = [r["kind"] for r in rs if r["role"] == "shipper" and r["matched"] == "True" and float(r["outcome"]) <= 0]
    for r in rs:
        if r["role"] == "shipper":
            p = float(r["outcome"])
            out.append((config, "shipper", r["kind"], 100 * p / sale))
            if r["matched"] == "True" and p > 0:
                cand[r["kind"]] = (sale - goods - p) - mu[r["kind"]]
    for r in rs:
        if r["role"] != "carrier": continue
        mg = float(r["outcome"])
        if r["matched"] != "True":
            out.append((config, "carrier", r["kind"], 0.0)); continue
        carriers_matched += 1
        hit = [s for s, v in cand.items() if abs(v - mg) < 1e-6]
        if not hit and len(declined) == 1:  # rate unrecoverable from a declined shipper's 0 profit
            hit = declined; fallback_declined[0] += 1
        if len(hit) != 1:
            unrecovered += 1; continue
        out.append((config, "carrier", r["kind"], 100 * mg / mu[hit[0]]))
print(f"matched carriers: {carriers_matched}, unrecovered: {unrecovered}, via declined shipper: {fallback_declined[0]}")

g = defaultdict(list)
for c, role, k, v in out: g[(c, role, k)].append(v)
rng = np.random.default_rng(0)
with open("data/validation/pct_outcomes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config","role","arch","n","mean_pct","ci_lo","ci_hi"])
    for (c, role, k), v in sorted(g.items()):
        a = np.array(v); bs = rng.choice(a, (5000, len(a))).mean(1)
        w.writerow([c, role, LBL[k], len(a), round(a.mean(), 2), round(np.quantile(bs, .025), 2), round(np.quantile(bs, .975), 2)])
        print(c, role, LBL[k], len(a), round(a.mean(), 2))
