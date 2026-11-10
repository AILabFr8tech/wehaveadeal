#!/usr/bin/env bash
# Trains CarrierRL2Fixed/ShipperRL2Fixed in small chunks, resuming from the
# last successfully-saved chunk if a chunk's subprocess crashes (this
# machine has shown intermittent native crashes in long-running
# torch/stable-baselines3 processes on our training machine).
# A crashed chunk just gets retried from the last checkpoint; total progress
# is never lost.
#
# Usage (from the repo root):  bash paper/scripts/train_rl2fixed_resilient.sh carrier [timesteps] [chunk]
# The result goes to paper/models/; the paper's weights ship in
# wehaveadeal/baselines_data/rl/ and are what CarrierRL2Fixed/ShipperRL2Fixed load by default.
set -u
ROLE="$1"
TOTAL_TIMESTEPS="${2:-300000}"
CHUNK_SIZE="${3:-20000}"
MAX_RETRIES_PER_CHUNK=15
mkdir -p paper/models
OUT="paper/models/${ROLE}_rl2fixed_model.zip"
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHON="${PYTHON:-python}"

done_steps=0
chunk_num=0
while [ "$done_steps" -lt "$TOTAL_TIMESTEPS" ]; do
  chunk_num=$((chunk_num + 1))
  remaining=$((TOTAL_TIMESTEPS - done_steps))
  this_chunk=$CHUNK_SIZE
  if [ "$remaining" -lt "$CHUNK_SIZE" ]; then
    this_chunk=$remaining
  fi

  resume_flag=""
  if [ "$done_steps" -gt 0 ]; then
    resume_flag="--resume-from $OUT"
  fi

  attempt=0
  ok=0
  while [ "$attempt" -lt "$MAX_RETRIES_PER_CHUNK" ]; do
    attempt=$((attempt + 1))
    echo "=== chunk $chunk_num (done=$done_steps, +$this_chunk) attempt $attempt/$MAX_RETRIES_PER_CHUNK ==="
    "$PYTHON" -u -m wehaveadeal.rl.train_league --role "$ROLE" --timesteps "$this_chunk" --out "$OUT" $resume_flag
    code=$?
    if [ "$code" -eq 0 ]; then
      ok=1
      break
    fi
    echo "=== chunk $chunk_num attempt $attempt FAILED (exit $code), retrying ==="
  done

  if [ "$ok" -ne 1 ]; then
    echo "=== chunk $chunk_num exhausted $MAX_RETRIES_PER_CHUNK retries, giving up ==="
    exit 1
  fi

  done_steps=$((done_steps + this_chunk))
  echo "=== chunk $chunk_num done, total trained: $done_steps/$TOTAL_TIMESTEPS ==="
done

echo "=== ALL DONE: $ROLE trained to $done_steps timesteps, model at $OUT ==="
