#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT_DIR/configs/mvtec_flow.yml}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/results/mvtec_flow/flow_epoch_0300.pth}"
SWEEP_DIR="$PROJECT_DIR/results/mvtec_flow/eval_sweep"
LOG_DIR="$SWEEP_DIR/logs"
TMP_CFG_DIR="$SWEEP_DIR/tmp_configs"
SUMMARY_TSV="$SWEEP_DIR/sweep_summary.tsv"
mkdir -p "$LOG_DIR" "$TMP_CFG_DIR"
if [ ! -f "$SUMMARY_TSV" ]; then
  printf '%s\n' 'Operator	Probe_T	Sigma	Curvature_Mode	Seed	I-AUROC	I-AP	P-AUROC	P-AP	AU-PRO	mAD	OutputFile' > "$SUMMARY_TSV"
fi

run_eval() {
  local gpu="$1" op="$2" probe_t="$3" sigma="$4" curv_mode="$5" seed="$6"
  local tag="${op}_t${probe_t}_sig${sigma}_mode${curv_mode}_s${seed}"
  local tmp_cfg="$TMP_CFG_DIR/cfg_gpu${gpu}_${tag}.yml"
  local out_json="$SWEEP_DIR/eval_${tag}.json"
  local log_file="$LOG_DIR/eval_${tag}.log"
  if [ -f "$out_json" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    echo "[GPU ${gpu}] SKIP ${tag}"
    return 0
  fi
  echo "[GPU ${gpu}] START ${tag}"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$tmp_cfg" "$out_json" "$op" "$probe_t" "$sigma" "$curv_mode" "$seed" <<'PY'
import sys
import yaml
base_config, tmp_cfg, out_json, op, probe_t, sigma, curv_mode, seed = sys.argv[1:]
with open(base_config, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["evaluation"].update(
    operator=op, probe_t=float(probe_t), gaussian_sigma=float(sigma),
    curvature_mode=curv_mode, anchor_seed=int(seed), output=out_json,
)
with open(tmp_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle)
PY
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$PROJECT_DIR/eval.py" \
      --checkpoint "$CHECKPOINT" --config "$tmp_cfg" --operator "$op" \
      --output "$out_json" >"$log_file" 2>&1; then
    "$PYTHON_BIN" - "$out_json" "$SUMMARY_TSV" "$op" "$probe_t" "$sigma" "$curv_mode" "$seed" <<'PY'
import json
import sys
out_json, summary_tsv, op, probe_t, sigma, curv_mode, seed = sys.argv[1:]
with open(out_json, "r", encoding="utf-8") as handle:
    data = json.load(handle)
result = data["results"].get(op)
if result is None and op == "both":
    result = data["results"]["angular"]
macro = result["macro_average"]
values = [op, probe_t, sigma, curv_mode, seed,
          f'{macro["I-AUROC"]:.6f}', f'{macro["I-AP"]:.6f}',
          f'{macro["P-AUROC"]:.6f}', f'{macro["P-AP"]:.6f}',
          f'{macro["AU-PRO"]:.6f}', f'{macro["mAD"]:.6f}', out_json]
with open(summary_tsv, "a", encoding="utf-8") as handle:
    handle.write("\t".join(values) + "\n")
PY
    echo "[GPU ${gpu}] DONE ${tag}"
    rm -f "$tmp_cfg"
  else
    status=$?
    echo "[GPU ${gpu}] FAIL ${tag} exit=${status}; see ${log_file}" >&2
    rm -f "$tmp_cfg"
    return "$status"
  fi
}

worker_gpu4() {
  local failed=0
  run_eval 4 angular 0.4 1.0 normalized_change 59 || failed=1
  run_eval 4 angular 0.4 1.5 normalized_change 59 || failed=1
  run_eval 4 angular 0.4 2.0 normalized_change 59 || failed=1
  return "$failed"
}
worker_gpu5() {
  local failed=0
  run_eval 5 angular 0.1 1.5 normalized_change 59 || failed=1
  run_eval 5 angular 0.2 1.5 normalized_change 59 || failed=1
  run_eval 5 angular 0.3 1.5 normalized_change 59 || failed=1
  return "$failed"
}
worker_gpu6() {
  local failed=0
  run_eval 6 curvature 0.2 1.5 normalized_change 59 || failed=1
  run_eval 6 curvature 0.4 1.5 normalized_change 59 || failed=1
  run_eval 6 curvature 0.2 1.5 perpendicular 59 || failed=1
  run_eval 6 curvature 0.4 1.5 perpendicular 59 || failed=1
  return "$failed"
}
worker_gpu7() {
  local failed=0
  run_eval 7 angular 0.2 1.5 normalized_change 42 || failed=1
  run_eval 7 angular 0.2 1.5 normalized_change 100 || failed=1
  run_eval 7 angular 0.2 1.5 normalized_change 2024 || failed=1
  return "$failed"
}

echo "Starting parallel eval sweep on physical GPUs 4,5,6,7"
echo "Checkpoint: $CHECKPOINT"
echo "Results:    $SWEEP_DIR"
worker_gpu4 & pid4=$!
worker_gpu5 & pid5=$!
worker_gpu6 & pid6=$!
worker_gpu7 & pid7=$!
failed=0
wait "$pid4" || failed=1
wait "$pid5" || failed=1
wait "$pid6" || failed=1
wait "$pid7" || failed=1
if [ "$failed" -ne 0 ]; then
  echo "Sweep completed with failures; inspect $LOG_DIR" >&2
  exit 1
fi
echo "Sweep complete. Summary: $SUMMARY_TSV"
