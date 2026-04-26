#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="pandoc-paper:local"

cd "$ROOT_DIR"

echo "[build] Building container image: $IMAGE_NAME"
docker build -f Dockerfile.pandoc -t "$IMAGE_NAME" .

echo "[build] Rendering main manuscript DOCX"
docker run --rm -v "$ROOT_DIR:/work" -w /work "$IMAGE_NAME" \
  digital_discovery_paper.md \
  --filter pandoc-citeproc \
  --bibliography references.bib \
  --csl royal-society-of-chemistry-with-titles.csl \
  -o TAS-AI_Digital_Discovery.docx

echo "[build] Rendering supplementary DOCX"
docker run --rm -v "$ROOT_DIR:/work" -w /work "$IMAGE_NAME" \
  TAS-AI_Digital_Discovery_SI.md \
  -o TAS-AI_Digital_Discovery_SI.docx

echo "[build] Done"
ls -lh TAS-AI_Digital_Discovery.docx TAS-AI_Digital_Discovery_SI.docx
