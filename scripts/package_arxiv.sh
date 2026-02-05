#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
MANUSCRIPT_DIR="$REPO_DIR/paper"

MAIN_TEX_INPUT="${1:-agency.tex}"
OUTPUT_PATH_INPUT="${2:-$MANUSCRIPT_DIR/arxiv_upload.tar.gz}"

if [[ "$MAIN_TEX_INPUT" = /* ]]; then
  MAIN_TEX="$MAIN_TEX_INPUT"
else
  MAIN_TEX="$MANUSCRIPT_DIR/$MAIN_TEX_INPUT"
fi

if [[ "$OUTPUT_PATH_INPUT" = /* ]]; then
  OUTPUT_PATH="$OUTPUT_PATH_INPUT"
else
  OUTPUT_PATH="$REPO_DIR/$OUTPUT_PATH_INPUT"
fi

if [[ ! -f "$MAIN_TEX" ]]; then
  echo "ERROR: main tex file not found: $MAIN_TEX" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mapfile -t TEX_FILES < <(
  python3 - "$MANUSCRIPT_DIR" "$MAIN_TEX" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
main = Path(sys.argv[2]).resolve()

if not main.exists():
    print(f"ERROR: main tex does not exist: {main}", file=sys.stderr)
    sys.exit(1)

def strip_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        buf = []
        i = 0
        escaped = False
        while i < len(line):
            ch = line[i]
            if ch == '%' and not escaped:
                break
            if ch == '\\':
                escaped = not escaped
            else:
                escaped = False
            buf.append(ch)
            i += 1
        out_lines.append(''.join(buf))
    return '\n'.join(out_lines)

seen = set()
files = set()

tex_re = re.compile(r"\\(?:input|include)\{([^}]+)\}")
graphics_re = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
bib_re = re.compile(r"\\bibliography\{([^}]+)\}")
addbib_re = re.compile(r"\\addbibresource\{([^}]+)\}")


def resolve_candidates(base: Path, name: str, exts):
    name = name.strip()
    p = Path(name)
    if p.suffix:
        return [ (base / p).resolve() ]
    return [ (base / (name + ext)).resolve() for ext in exts ]


def parse_file(path: Path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    files.add(path)

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    text = strip_comments(text)

    for m in tex_re.findall(text):
        candidates = resolve_candidates(path.parent, m, [".tex"])
        if not any(c.exists() for c in candidates):
            candidates = resolve_candidates(root, m, [".tex"])
        for cand in candidates:
            if cand.exists():
                parse_file(cand)
                break

    for m in graphics_re.findall(text):
        candidates = resolve_candidates(path.parent, m, [".pdf", ".png", ".jpg", ".jpeg", ".eps"])
        if not any(c.exists() for c in candidates):
            candidates = resolve_candidates(root, m, [".pdf", ".png", ".jpg", ".jpeg", ".eps"])
        for cand in candidates:
            if cand.exists():
                files.add(cand)
                break

    for m in bib_re.findall(text):
        for part in m.split(','):
            candidates = resolve_candidates(path.parent, part, [".bib"])
            if not any(c.exists() for c in candidates):
                candidates = resolve_candidates(root, part, [".bib"])
            for cand in candidates:
                if cand.exists():
                    files.add(cand)
                    break

    for m in addbib_re.findall(text):
        candidates = resolve_candidates(path.parent, m, [""])
        if not any(c.exists() for c in candidates):
            candidates = resolve_candidates(root, m, [""])
        for cand in candidates:
            if cand.exists():
                files.add(cand)
                break

parse_file(main)

for f in sorted(files):
    try:
        f.relative_to(root)
    except ValueError:
        print(f"WARNING: skipping file outside manuscript dir: {f}", file=sys.stderr)
        continue
    print(str(f))
PY
)

# Include the main .bbl if present (helps arXiv compile without BibTeX).
MAIN_BBL="${MAIN_TEX%.tex}.bbl"
if [[ -f "$MAIN_BBL" ]]; then
  TEX_FILES+=("$MAIN_BBL")
fi

# Include local style/class/bibliography files if present.
while IFS= read -r style_file; do
  TEX_FILES+=("$style_file")
done < <(find "$MANUSCRIPT_DIR" -maxdepth 2 -type f \( -name "*.sty" -o -name "*.cls" -o -name "*.bst" -o -name "*.bbx" -o -name "*.cbx" \))

# De-duplicate
readarray -t UNIQUE_FILES < <(printf "%s\n" "${TEX_FILES[@]}" | awk 'NF' | sort -u)

# Copy files into temp dir preserving paths relative to manuscript.
for file in "${UNIQUE_FILES[@]}"; do
  rel="${file#$MANUSCRIPT_DIR/}"
  dest_dir="$TMP_DIR/$(dirname "$rel")"
  mkdir -p "$dest_dir"
  cp "$file" "$dest_dir/"
done

mkdir -p "$(dirname "$OUTPUT_PATH")"

tar -czf "$OUTPUT_PATH" -C "$TMP_DIR" .

echo "arXiv package created: $OUTPUT_PATH"
