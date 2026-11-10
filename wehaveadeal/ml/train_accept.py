"""Train the accept-vs-continue classifier that, together with
`wehaveadeal/ml/train.py`'s offer regressor, makes up `CarrierML2`/`ShipperML2`
(see `wehaveadeal/agents/carrier_ml2.py` for how the two models are combined).

Reuses `wehaveadeal.ml.collect`'s data unchanged and `wehaveadeal.ml.train`'s
`load_rows`/`split_by_seed` as-is -- same self-play data, same feature
schema, same seed-level split, just a different target column and a
different row filter:

  - Target: `accepted` (1 if the teacher accepted here, 0 if it countered).
  - Filter: only `accept_eligible == 1` rows -- a row where the opponent's
    offer wasn't even legal to accept isn't a real "accept vs. continue"
    example (see `wehaveadeal/ml/collect.py`'s module docstring).

Model choice mirrors the offer regressor: `HistGradientBoostingClassifier`
(same scikit-learn family, zero new dependencies), not a from-scratch
formula -- this is what replaces the hand-derived value-function comparison
`CarrierRV`/`CarrierML` use for the same decision (see
`wehaveadeal/agents/carrier_ml.py`'s module docstring for why ML1 kept that
comparison hardcoded instead of learning it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wehaveadeal.ml.features import FEATURE_NAMES
from wehaveadeal.ml.train import load_rows, split_by_seed


def main() -> None:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    parser = argparse.ArgumentParser(description="Train the accept-vs-continue classifier from self-play data")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--role", choices=["carrier", "shipper"], required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument(
        "--out", type=str, default=None,
        help="defaults to data/ml/<role>_ml2_accept_model.joblib -- load it with CarrierML2(accept_model_path=...)",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    X, _label, accepted, accept_eligible, seeds = load_rows(Path(args.data))
    eligible = accept_eligible == 1
    X, y, seeds = X[eligible], accepted[eligible], seeds[eligible]
    train_mask, val_mask = split_by_seed(seeds, args.val_frac, args.split_seed)
    print(f"rows: {len(y)} accept-eligible ({eligible.sum()}/{len(eligible)} of all rows), "
          f"{train_mask.sum()} train, {val_mask.sum()} val "
          f"({len(np.unique(seeds[train_mask]))} train seeds, {len(np.unique(seeds[val_mask]))} val seeds)")
    print(f"class balance -- accepted: {y.mean():.2%} of eligible rows")

    model = HistGradientBoostingClassifier(
        max_depth=args.max_depth, max_iter=args.max_iter, learning_rate=args.learning_rate, random_state=0
    )
    model.fit(X[train_mask], y[train_mask])

    train_pred = model.predict(X[train_mask])
    val_pred = model.predict(X[val_mask])
    train_proba = model.predict_proba(X[train_mask])[:, 1]
    val_proba = model.predict_proba(X[val_mask])[:, 1]

    train_acc = float(accuracy_score(y[train_mask], train_pred))
    val_acc = float(accuracy_score(y[val_mask], val_pred))
    # Naive baseline: always predict the majority class seen in training.
    # If val accuracy isn't meaningfully better than this, the classifier
    # hasn't learned anything beyond "the teacher usually does X".
    majority_class = int(round(y[train_mask].mean()))
    naive_acc = float(accuracy_score(y[val_mask], np.full(val_mask.sum(), majority_class)))
    try:
        train_auc = float(roc_auc_score(y[train_mask], train_proba))
        val_auc = float(roc_auc_score(y[val_mask], val_proba))
    except ValueError:
        # Only one class present in a split (can happen with few seeds) --
        # AUC is undefined then, not a bug.
        train_auc = val_auc = float("nan")

    print(f"train accuracy: {train_acc:.4f}   train AUC: {train_auc:.4f}")
    print(f"val accuracy:   {val_acc:.4f}   val AUC:   {val_auc:.4f}")
    print(f"naive baseline val accuracy: {naive_acc:.4f}  (always predict majority class)")

    out = Path(args.out) if args.out else Path("data/ml") / f"{args.role}_ml2_accept_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    meta = {
        "role": args.role,
        "teacher_kind": args.teacher,
        "feature_names": list(FEATURE_NAMES),
        "n_rows": len(y),
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
        "accepted_frac": float(y.mean()),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "naive_val_accuracy": naive_acc,
        "model_class": type(model).__name__,
        "hyperparams": {"max_depth": args.max_depth, "max_iter": args.max_iter, "learning_rate": args.learning_rate},
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote model to {out}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
