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
  --exclude 'error.txt' \
  --exclude 'RL Qwen Model Analysis.docx' \
  "$REPO_DIR/" "$OUT_DIR/"

for folder_name in staged_rl tests; do
  if [[ -d "$OUT_DIR/$folder_name" ]]; then
    tar -cf "$OUT_DIR/${folder_name}.tar" -C "$OUT_DIR" "$folder_name"
    rm -rf "$OUT_DIR/$folder_name"
  fi
done

echo "Prepared Kaggle upload directory: $OUT_DIR"
