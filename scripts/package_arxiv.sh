#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
BUILD_DIR="$PAPER_DIR/build"
STAGE_DIR="$BUILD_DIR/arxiv_source_staging"
ZIP_PATH="$BUILD_DIR/arxiv_source_upload.zip"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/sections" "$STAGE_DIR/figures" "$STAGE_DIR/bib"

cp "$PAPER_DIR/agency.tex" "$STAGE_DIR/agency.tex"
cp "$PAPER_DIR/preamble.tex" "$STAGE_DIR/preamble.tex"
cp "$PAPER_DIR/bib/refs.bib" "$STAGE_DIR/bib/refs.bib"

if [[ -f "$PAPER_DIR/agency.bbl" ]]; then
  cp "$PAPER_DIR/agency.bbl" "$STAGE_DIR/agency.bbl"
elif [[ -f "$BUILD_DIR/agency.bbl" ]]; then
  cp "$BUILD_DIR/agency.bbl" "$STAGE_DIR/agency.bbl"
fi

if compgen -G "$PAPER_DIR/sections/*.tex" > /dev/null; then
  cp "$PAPER_DIR/sections/"*.tex "$STAGE_DIR/sections/"
fi

for ext in png jpg pdf; do
  if compgen -G "$PAPER_DIR/figures/*.$ext" > /dev/null; then
    cp "$PAPER_DIR/figures/"*.$ext "$STAGE_DIR/figures/"
  fi
done

if compgen -G "$PAPER_DIR/figures/*.eps" > /dev/null; then
  echo "[package_arxiv] ERROR: EPS figures detected; convert to PDF/PNG." >&2
  exit 4
fi

find "$STAGE_DIR" -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" | xargs -r rm -f

rm -f "$ZIP_PATH"
(
  cd "$STAGE_DIR"
  zip -r "$ZIP_PATH" . >/dev/null
)

echo "[package_arxiv] Wrote $ZIP_PATH"
