"""Train an imitation-learning regressor on data from `wehaveadeal.ml.collect`,
and package it (weights + feature schema + metadata) for
`wehaveadeal.agents.carrier_ml.CarrierML` / `wehaveadeal.agents.shipper_ml.ShipperML`
to load at inference time.

Model choice: a small gradient-boosted tree ensemble
(`HistGradientBoostingRegressor`, scikit-learn's LightGBM-alike) over the
engineered feature vector -- not a sequence model. The turn history here is
at most 6 points per side, already collapsed into a handful of summary
features (trend slope/intercept, band positions) by `wehaveadeal.ml.features`,
so there's no sequence left for an RNN/Transformer to exploit; a tabular
GBM is the standard strong baseline for exactly this shape of problem
(heterogeneous small-to-medium tabular features, modest data) and keeps
some interpretability via feature importances, unlike a deep net.

The train/validation split is by SEED, not by row: turns within the same
table (and tables within the same negotiation) are highly correlated, so a
row-level split would leak information across the split.

`load_rows`/`split_by_seed` are reused as-is by `wehaveadeal/ml/train_accept.py`
(the accept-vs-continue classifier that, together with this regressor,
makes up `CarrierML2`/`ShipperML2` -- see `wehaveadeal/agents/carrier_ml2.py`).
This script itself only ever trains on `accepted == 0` rows (an ACCEPT row
has no "offer" to regress against -- see `wehaveadeal/ml/collect.py`), so it
keeps working unchanged even though the CSV it reads now also contains the
accept-decision rows the classifier needs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from wehaveadeal.ml.features import FEATURE_NAMES


def load_rows(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, label, accepted, accept_eligible, seeds) for every row in
    the CSV. `accepted`/`accept_eligible` default to 0 when reading an
    older CSV written before those columns existed, so this still loads
    pre-ML2 datasets without special-casing."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=float)
    label = np.array([float(r["label"]) for r in rows], dtype=float)
    accepted = np.array([int(r.get("accepted") or 0) for r in rows], dtype=int)
    accept_eligible = np.array([int(r.get("accept_eligible") or 0) for r in rows], dtype=int)
    seeds = np.array([int(r["seed"]) for r in rows], dtype=int)
    return X, label, accepted, accept_eligible, seeds


def split_by_seed(seeds: np.ndarray, val_frac: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(seeds)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_seeds = set(unique[:n_val].tolist())
    is_val = np.array([s in val_seeds for s in seeds])
    return ~is_val, is_val


def main() -> None:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    parser = argparse.ArgumentParser(description="Train an imitation-learning model from self-play data")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--role", choices=["carrier", "shipper"], required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", type=str, default=None, help="defaults to data/ml/<role>_ml_model.joblib -- load it with CarrierML(model_path=...) / ShipperML(model_path=...)")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    X, label, accepted, _accept_eligible, seeds = load_rows(Path(args.data))
    offer_mask = accepted == 0  # this regressor only ever trains on "what did the teacher offer" rows
    X, y, seeds = X[offer_mask], label[offer_mask], seeds[offer_mask]
    train_mask, val_mask = split_by_seed(seeds, args.val_frac, args.split_seed)
    print(f"rows: {len(y)} total, {train_mask.sum()} train, {val_mask.sum()} val "
          f"({len(np.unique(seeds[train_mask]))} train seeds, {len(np.unique(seeds[val_mask]))} val seeds)")

    model = HistGradientBoostingRegressor(
        max_depth=args.max_depth, max_iter=args.max_iter, learning_rate=args.learning_rate, random_state=0
    )
    model.fit(X[train_mask], y[train_mask])

    train_mae = float(mean_absolute_error(y[train_mask], model.predict(X[train_mask])))
    val_mae = float(mean_absolute_error(y[val_mask], model.predict(X[val_mask])))
    # A naive baseline (predict the mean training label) puts these numbers
    # in context: if val_mae isn't meaningfully better than this, the model
    # hasn't learned anything beyond "the teacher usually offers around X".
    naive_mae = float(mean_absolute_error(y[val_mask], np.full(val_mask.sum(), y[train_mask].mean())))
    print(f"train MAE (band units, 0-1): {train_mae:.4f}")
    print(f"val MAE   (band units, 0-1): {val_mae:.4f}")
    print(f"naive baseline val MAE:       {naive_mae:.4f}  (predict-the-mean)")

    out = Path(args.out) if args.out else Path("data/ml") / f"{args.role}_ml_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    meta = {
        "role": args.role,
        "teacher_kind": args.teacher,
        "feature_names": list(FEATURE_NAMES),
        "n_rows": len(y),
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
        "train_mae": train_mae,
        "val_mae": val_mae,
        "naive_val_mae": naive_mae,
        "model_class": type(model).__name__,
        "hyperparams": {"max_depth": args.max_depth, "max_iter": args.max_iter, "learning_rate": args.learning_rate},
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote model to {out}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
