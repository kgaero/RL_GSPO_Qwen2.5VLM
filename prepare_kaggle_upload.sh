#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/rl-gspo-qwen2-5vlm-kaggle-dataset}"

mkdir -p "$OUT_DIR"

rsync -av \
  --exclude .git \
  --exclude outputs \
  --exclude outputs_staged \
  --exclude grpo_lora \
  --exclude grpo_eval_outputs \
  --exclude grpo_trainer_lora_model \
  --exclude unsloth_compiled_cache \
  --exclude __pycache__ \
  --exclude '*.safetensors' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.onnx' \
  --exclude 'RL Qwen Model Analysis.docx' \
  "$REPO_DIR/" "$OUT_DIR/"

echo "Prepared Kaggle upload directory: $OUT_DIR"
