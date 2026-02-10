# TechRxiv QA Report

## Build Verification

| Item | Status |
|------|--------|
| Build command | `cd paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error agency.tex` |
| Output | `agency.pdf` (27 pages, 561 KB) |
| Undefined references | 0 |
| Missing citations | 0 |
| Meaningful LaTeX warnings | 1 (float too large for page by 26pt --- cosmetic only) |

## Lean Build

| Item | Status |
|------|--------|
| Build command | `cd lean && lake build` |
| Result | SUCCESS (7887 jobs) |
| Warnings | 3 style lint warnings (unused DecidableEq hypothesis, simpa vs simp) --- cosmetic only |

## Artifact Audit

| Item | Status |
|------|--------|
| Audit command | `python scripts/audit_results.py --strict` |
| Result | 11 artifacts checked, 0 errors, 0 warnings |

## Figures Verified Present

| Figure | File | Section | Status |
|--------|------|---------|--------|
| fig_packaging_ring.png | paper/figures/ | Exhibit: packaging | Present (31 KB) |
| fig_protocol_horizon.png | paper/figures/ | Exhibit: holonomy | Present (28 KB) |
| fig_sweep_K.png | paper/figures/ | Exhibit: sweep | Present (20 KB) |
| fig_sweep_E.png | paper/figures/ | Exhibit: sweep | Present (20 KB) |
| fig_learning_theta.png | paper/figures/ | Exhibit: learning | Present (20 KB) |

## Tables Verified

| Table | Label | Source | Status |
|-------|-------|--------|--------|
| Dictionary | tab:dictionary | sections/02_dictionary.tex | Compiles, referenced |
| Null regimes | tab:nulls | sections/05_exhibit_nulls.tex | Compiles, referenced |
| Ablations | tab:ablations | generated/ablations_summary.tex | Compiles, referenced |

## Generated Assets Verified

| File | Status |
|------|--------|
| paper/generated/numbers.json | Present (1.4 KB) |
| paper/generated/ablations_summary.tex | Present (390 B) |
| paper/generated/holonomy_witness.tex | Present (253 B) |

## Changes Made (high level)

### Cover page template (matching DE paper)
- Added `\usepackage{orcidlink}` to preamble
- Updated author to inline ORCID: `\orcidlink{0009-0009-7659-5964}`
- Emptied `\affiliation{}` (moved to footer)
- Added version/revision date line with Zenodo v1 DOI
- Updated footer to match DE template: affiliation, email, DOI, copyright, CC-BY 4.0, preprint disclaimer

### Abstract rewrite (scope-framing for CS/tech)
- Rewritten to lead with "reproducible computational framework" and concrete metrics
- Added theory disambiguation sentence ("theory is used in the SBT technical sense...")
- Added mention of Lean anchor and strict audit mode
- Preserved all quantitative claims and caution sentence

### Keywords update
- Expanded from 5 to 8 keywords, targeting CS/tech discoverability
- Added: viability kernel, controlled invariance, channel capacity, Markov decision processes, reproducible artifacts

### Introduction additions
- Added "Research Article" disambiguation sentence
- Added "Computational contributions" paragraph listing 5 concrete CS deliverables
- Added framing sentence tying to control theory, information theory, and formal methods

### Discussion/conclusion additions
- Renamed "Closing" to "Conclusion"
- Added full Declarations section (corresponding author, competing interests, funding, ethics, data availability, code availability, author contributions, AI/LLM disclosure)

### Language pass
- No instances of "slogan", "canonical" (used philosophically), or "real" (used problematically) found
- All figure/table references verified intact

## Remaining Human-Must-Fill Items

None. All metadata (ORCID, affiliation, DOI, license) is populated.

## Security / Sanitization Check

- No API keys, tokens, or secrets found in paper source
- GitHub Actions workflow tokens are standard `${{ secrets.GITHUB_TOKEN }}` (not real secrets)
- No private dataset paths or cached outputs with sensitive metadata
- Upload is PDF-only; code/data referenced via external links (GitHub + Zenodo DOI)

## Original Files

Pre-edit originals archived in `archive/zenodo_preprint/` (committed as `650c479`).
