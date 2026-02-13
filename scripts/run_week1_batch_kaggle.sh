#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-core}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PROFILE}" == "full" ]]; then
  CONFIGS=(
    "configs/cloud/cdgp_week1_batch_cifar10_random_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_entropy_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w05_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w08_kaggle.json"
  )
else
  CONFIGS=(
    "configs/cloud/cdgp_week1_batch_cifar10_random_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_entropy_kaggle.json"
    "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle.json"
  )
fi

echo "Python: ${PYTHON_BIN}"
${PYTHON_BIN} - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi
  echo ""
  echo "=== Running ${config} ==="
  PYTHONPATH=src ${PYTHON_BIN} src/main.py --mode phase1b --config "${config}"
done

echo ""
echo "Week 1 Kaggle batch (${PROFILE}) completed."
