#!/usr/bin/env bash
# The whole training pipeline, resumable, in one work folder:
#   1. the modal bank, the validation shard and the benchmark (built once, kept);
#   2. training, one process per GPU (resumes from the folder's last.pt);
#   3. the ONNX export of the best checkpoint, and its benchmark report.
#
# Everything lives under $DISPICK_WORK, which should be persistent storage: a mounted disk,
# a Cloud Storage FUSE mount on Vertex AI (/gcs/<bucket>/...), or SageMaker's checkpoint
# folder (/opt/ml/checkpoints, synced to S3). Re-running the script after an interruption
# (a spot instance reclaimed) carries on where it stopped.
#
# Settings (environment variables):
#   DISPICK_WORK        work folder (default /work)
#   DISPICK_CONFIG      training configuration (default configs/train_cloud.yaml)
#   DISPICK_RUN         run name, the folder under runs/ (default cloud)
#   DISPICK_BANK_MODELS models in the bank (default 50000)
#   DISPICK_VALIDATION  validation examples (default 4096)
#   DISPICK_BENCHMARK   benchmark images (default 2000)
#   DISPICK_GPUS        processes (default: every visible GPU, else 1 on the CPU)
#   DISPICK_WORKERS     CPU workers for the data builds (default: all cores)
#   DISPICK_TRAIN_ARGS  extra `dispick train` arguments, e.g. "--set optim.steps=200000"
#   DISPICK_LOCAL       local disk the data are copied to for training (default
#                       /tmp/dispick-data): every image reads the bank, which a bucket mount
#                       would make slow
set -euo pipefail

if [[ "${1:-}" == "train" ]]; then shift; fi  # SageMaker's convention

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${DISPICK_WORK:-/work}"
config="${DISPICK_CONFIG:-${here}/configs/train_cloud.yaml}"
run="${DISPICK_RUN:-cloud}"
models="${DISPICK_BANK_MODELS:-50000}"
validation_count="${DISPICK_VALIDATION:-4096}"
benchmark_count="${DISPICK_BENCHMARK:-2000}"
workers="${DISPICK_WORKERS:-$(nproc)}"
if [[ -n "${DISPICK_GPUS:-}" ]]; then
  gpus="${DISPICK_GPUS}"
else
  gpus="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
fi

# The shards must be on the grid the network trains on.
read -r grid_f grid_v < <(python -c 'import sys, yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
print(*config.get("data", {}).get("grid", [256, 256]))' "${config}")
data="${work}/data"
output="${work}/runs/${run}"
mkdir -p "${data}" "${output}"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "work folder ${work}, ${gpus} process(es), ${workers} CPU workers"
if [[ ! -f "${data}/bank.h5" ]]; then
  log "building the modal bank (${models} models)"
  dispick bank build --config "${here}/configs/bank.yaml" --models "${models}" \
    --out "${data}/bank.h5" --workers "${workers}"
fi
if [[ ! -f "${data}/validation.h5" ]]; then
  log "building the validation shard (${validation_count} examples)"
  dispick data shard --bank "${data}/bank.h5" --out "${data}/validation.h5" \
    --count "${validation_count}" --seed 1001 --grid "${grid_f}" "${grid_v}" --workers "${workers}"
fi
if [[ ! -f "${data}/benchmark.h5" ]]; then
  log "building the benchmark (${benchmark_count} images)"
  dispick data benchmark --bank "${data}/bank.h5" --out "${data}/benchmark.h5" \
    --count "${benchmark_count}" --seed 2002 --workers "${workers}"
fi

local_data="${DISPICK_LOCAL:-/tmp/dispick-data}"
if [[ "$(realpath "${local_data}" 2>/dev/null)" != "$(realpath "${data}")" ]]; then
  mkdir -p "${local_data}"
  cp -u "${data}/bank.h5" "${data}/validation.h5" "${local_data}/"
fi

log "training (${config})"
# shellcheck disable=SC2086  # DISPICK_TRAIN_ARGS is a list of arguments
torchrun --standalone --nproc-per-node="${gpus}" -m dispick.cli train --config "${config}" \
  --set "output=${output}" --set "data.bank=${local_data}/bank.h5" \
  --set "data.validation=${local_data}/validation.h5" ${DISPICK_TRAIN_ARGS:-}

checkpoint="${output}/best.pt"
[[ -f "${checkpoint}" ]] || checkpoint="${output}/last.pt"
log "exporting ${checkpoint}"
dispick export --checkpoint "${checkpoint}" --out "${output}/dispick.onnx"
log "benchmarking"
dispick evaluate --model "${output}/dispick.onnx" --benchmark "${data}/benchmark.h5" \
  --out "${output}/report" --baselines --device cpu

# SageMaker uploads /opt/ml/model when the job ends.
if [[ -d /opt/ml/model ]]; then
  cp "${output}/dispick.onnx" "${output}/dispick.json" /opt/ml/model/
  cp -r "${output}/report" /opt/ml/model/report
fi
log "done: ${output}/dispick.onnx"
