#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
BUILD_DIR="$PAPER_DIR/build"
STAGE_DIR="$BUILD_DIR/hal_source_staging"
ZIP_PATH="$BUILD_DIR/hal_source_upload.zip"
MAIN_TEX="$PAPER_DIR/agency.tex"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/sections" "$STAGE_DIR/figures" "$STAGE_DIR/bib"

# Absolute path guard for includegraphics
if rg -n "\\\\includegraphics\\{/" "$MAIN_TEX" >/dev/null 2>&1; then
  echo "[make_hal_source_zip] ERROR: absolute path found in \\includegraphics{}." >&2
  exit 2
fi

# Ensure .bbl exists (arXiv/HAL does not run BibTeX).
if ! [[ -f "$BUILD_DIR/agency.bbl" ]]; then
  echo "[make_hal_source_zip] agency.bbl missing; compiling to generate it..." >&2
  (cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" agency.tex)
  (cd "$BUILD_DIR" && bibtex agency)
fi

if ! [[ -f "$BUILD_DIR/agency.bbl" ]]; then
  echo "[make_hal_source_zip] ERROR: agency.bbl not generated." >&2
  exit 3
fi

# Core sources
cp "$MAIN_TEX" "$STAGE_DIR/agency.tex"
cp "$PAPER_DIR/preamble.tex" "$STAGE_DIR/preamble.tex"
cp "$PAPER_DIR/bib/refs.bib" "$STAGE_DIR/bib/refs.bib"
cp "$BUILD_DIR/agency.bbl" "$STAGE_DIR/agency.bbl"

# Sections
if compgen -G "$PAPER_DIR/sections/*.tex" > /dev/null; then
  cp "$PAPER_DIR/sections/"*.tex "$STAGE_DIR/sections/"
fi

# Figures (png/jpg/pdf only)
for ext in png jpg pdf; do
  if compgen -G "$PAPER_DIR/figures/*.$ext" > /dev/null; then
    cp "$PAPER_DIR/figures/"*.$ext "$STAGE_DIR/figures/"
  fi
done

# Reject EPS figures
if compgen -G "$PAPER_DIR/figures/*.eps" > /dev/null; then
  echo "[make_hal_source_zip] ERROR: EPS figures detected; convert to PDF/PNG." >&2
  exit 4
fi

# Remove build artifacts if any slipped in
find "$STAGE_DIR" -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" | xargs -r rm -f

rm -f "$ZIP_PATH"
(cd "$STAGE_DIR" && zip -r "$ZIP_PATH" . >/dev/null)

echo "[make_hal_source_zip] Wrote $ZIP_PATH"
