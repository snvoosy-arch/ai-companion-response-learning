#!/usr/bin/env bash
set -euo pipefail

ROOT="<repo>"
BOT_DIR="$ROOT/predictive-discord-bot"
VENV_DIR="${BLACK_TRAINING_VENV:-$BOT_DIR/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"

START_MODEL="${START_MODEL:-$ROOT/models/candidates/black/intent/modernbert_meaning_gold_direct_v26_unseeded_context_reweighted_20260502}"
TRAIN_DATASET="${TRAIN_DATASET:-$BOT_DIR/data/meaning/black_meaning_gold_direct_v27_fullreview_pass_mix_20260504_train.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-$BOT_DIR/data/meaning/black_meaning_gold_direct_v27_fullreview_pass_mix_20260504_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/candidates/black/intent/modernbert_meaning_gold_direct_v27_fullreview_pass_mix_20260504}"
REPORT_OUT="${REPORT_OUT:-$BOT_DIR/reports/modernbert_meaning_gold_direct_v27_fullreview_pass_mix_20260504_train_report.json}"
TRAIN_LOG_PATH="${BLACK_V27_FULLREVIEW_PASS_MIX_LOG:-$BOT_DIR/reports/modernbert_meaning_gold_direct_v27_fullreview_pass_mix_20260504_train.log}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-192}"
SLOT_LOSS_WEIGHT="${SLOT_LOSS_WEIGHT:-0.5}"
SLOT_OUTSIDE_WEIGHT="${SLOT_OUTSIDE_WEIGHT:-0.05}"
SEED="${SEED:-20260504}"
LOG_EVERY="${LOG_EVERY:-25}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${START_MODEL}/meaning_head_config.json" ]]; then
  echo "start ModernBERT meaning model not found: ${START_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_DATASET}" ]]; then
  echo "train dataset not found: ${TRAIN_DATASET}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_DATASET}" ]]; then
  echo "eval dataset not found: ${EVAL_DATASET}" >&2
  exit 1
fi

mkdir -p "$(dirname "${REPORT_OUT}")" "$(dirname "${TRAIN_LOG_PATH}")" "${OUTPUT_DIR}"

cd "$ROOT"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

{
  echo "[black-v27-fullreview-pass-mix] start_model=${START_MODEL}"
  echo "[black-v27-fullreview-pass-mix] train=${TRAIN_DATASET}"
  echo "[black-v27-fullreview-pass-mix] eval=${EVAL_DATASET}"
  echo "[black-v27-fullreview-pass-mix] output=${OUTPUT_DIR}"
  echo "[black-v27-fullreview-pass-mix] epochs=${EPOCHS} lr=${LEARNING_RATE} max_length=${MAX_LENGTH}"
} | tee "${TRAIN_LOG_PATH}"

"${PYTHON_BIN}" "$BOT_DIR/scripts/train_black_meaning_modernbert.py" \
  --train "${TRAIN_DATASET}" \
  --eval "${EVAL_DATASET}" \
  --model-name-or-path "${START_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --report-out "${REPORT_OUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --max-length "${MAX_LENGTH}" \
  --slot-loss-weight "${SLOT_LOSS_WEIGHT}" \
  --slot-outside-weight "${SLOT_OUTSIDE_WEIGHT}" \
  --seed "${SEED}" \
  --log-every "${LOG_EVERY}" \
  --balanced-loss \
  --balanced-slot-loss \
  2>&1 | tee -a "${TRAIN_LOG_PATH}"
