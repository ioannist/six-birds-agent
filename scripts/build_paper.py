#!/usr/bin/env python3
"""Build the LaTeX paper into paper/build/agency.pdf."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    tex_path = paper_dir / "agency.tex"
    out_dir = paper_dir / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")

    if latexmk:
        cmd = [
            latexmk,
            "-pdf",
            "-g",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={out_dir}",
            str(tex_path.name),
        ]
        subprocess.run(cmd, check=True, cwd=paper_dir)
    elif pdflatex:
        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={out_dir}",
            str(tex_path.name),
        ]
        subprocess.run(cmd, check=True, cwd=paper_dir)
        subprocess.run(cmd, check=True, cwd=paper_dir)
    else:
        raise SystemExit("Missing LaTeX tools: latexmk or pdflatex is required.")

    pdf_path = out_dir / "agency.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"Build failed: {pdf_path} not found")

    flattener = shutil.which("latexpand") or shutil.which("texflatten")
    if not flattener:
        raise SystemExit(
            "Missing LaTeX flattener: install 'latexpand' (recommended) or 'texflatten'."
        )

    flat_path = out_dir / "agency_flattened.tex"
    if Path(flattener).name == "latexpand":
        cmd = [flattener, "-o", str(flat_path), str(tex_path.name)]
        subprocess.run(cmd, check=True, cwd=paper_dir)
    else:
        result = subprocess.run(
            [flattener, str(tex_path.name)],
            check=True,
            cwd=paper_dir,
            text=True,
            stdout=subprocess.PIPE,
        )
        flat_path.write_text(result.stdout, encoding="utf-8")

    print(f"Build succeeded: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
