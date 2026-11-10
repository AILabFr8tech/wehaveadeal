"""Train the linear regression used by examples/linreg_carrier.py.

1. Collect: the nine reference carriers play the nine reference shippers.
   A thin wrapper records what each carrier could see at every decision.
2. Label: each record gets the rate its table finally closed at, relative
   to the market mean (tables that never closed are dropped).
3. Fit: scikit-learn LinearRegression, evaluated on held-out market days.
4. Save: model.joblib next to this script, where the agent looks for it.

    python examples/linreg_carrier_model/train.py               # ~1 min
    python examples/linreg_carrier_model/train.py --rounds 100  # more data

The training seeds start at 10_000, well away from the benchmark's default
seed, so the model is never trained on the market days it is scored on.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from wehaveadeal import DecisionEngine
from wehaveadeal.simulate import CARRIER_KINDS, SHIPPER_KINDS, build_carrier, build_shipper, run_round

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # to import the agent's feature function
from linreg_carrier import FEATURE_NAMES, features  # noqa: E402


class Recorder(DecisionEngine):
    """Plays exactly like the wrapped agent and logs every state it sees."""

    def __init__(self, inner: DecisionEngine, log: list) -> None:
        super().__init__(inner.agent_id, frozen=True)
        self.inner, self.log = inner, log
        self.engine_name = inner.engine_name

    def next_move(self, state):
        self.log.append((state.negotiation_seed, state.table_id, state.refprice_mean, features(state)))
        return self.inner.next_move(state)


def collect(n_rounds: int, n_negotiations: int, base_seed: int):
    """Return X (features), y (closing rate / mean) and the round of each row."""
    X, y, groups = [], [], []
    shippers = {f"shipper-{k}-1": build_shipper(k, f"shipper-{k}-1") for k in SHIPPER_KINDS}
    carriers = {f"carrier-{k}-1": build_carrier(k, f"carrier-{k}-1") for k in CARRIER_KINDS}
    for r in range(n_rounds):
        log = []
        recorded = {cid: Recorder(c, log) for cid, c in carriers.items()}
        result = run_round(r, base_seed + r, shippers, recorded, n_negotiations=n_negotiations)
        closing = {
            (neg.seed, f"{t.shipper_id}::{t.carrier_id}"): t.rate
            for neg in result.negotiations for t in neg.tables
            if t.closed and t.rate is not None
        }
        for seed, table_id, mean, row in log:
            rate = closing.get((seed, table_id))
            if rate is not None:
                X.append(row)
                y.append(rate / mean)
                groups.append(r)
        print(f"  round {r + 1}/{n_rounds}: {len(X)} labelled decisions so far", end="\r", flush=True)
    print()
    return np.array(X), np.array(y), np.array(groups)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rounds", type=int, default=30, help="market rounds to simulate")
    p.add_argument("--negotiations", type=int, default=5, help="market days per round")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("model.joblib"))
    args = p.parse_args(argv)

    print(f"Collecting {args.rounds} rounds x {args.negotiations} days of reference play ...")
    X, y, groups = collect(args.rounds, args.negotiations, args.seed)

    # Hold out the last 20% of rounds (whole market days, so no leakage between train and test).
    cut = int(args.rounds * 0.8)
    train, test = groups < cut, groups >= cut
    model = LinearRegression().fit(X[train], y[train])
    pred = model.predict(X[test])

    print(f"\nTrain rows: {train.sum()}   test rows: {test.sum()}")
    print(f"Test R^2: {r2_score(y[test], pred):.3f}   "
          f"mean abs error: {100 * mean_absolute_error(y[test], pred):.2f}% of the market mean")
    print("Coefficients (closing rate / market mean):")
    print(f"  {'intercept':<15} {model.intercept_:+.4f}")
    for name, coef in zip(FEATURE_NAMES, model.coef_):
        print(f"  {name:<15} {coef:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
