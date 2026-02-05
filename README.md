# Six Birds: Agency Instantiation

This repository contains the **agency instantiation** for the paper:

> **To Throw a Stone with Six Birds: An Agent is a Theory Object**
>
> Archived at: https://zenodo.org/records/18439737
>
> DOI: https://doi.org/10.5281/zenodo.18439737

This paper is the agency-focused instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It defines agenthood as a theory object (a maintained package inside a layer), operationalizes viability, feasibility-gated empowerment, and packaging stability, and demonstrates the resulting claims in a minimal finite ring-world substrate.

## What this repository provides

The agency instantiation implements:

- **Finite kernel substrate**: controlled stochastic kernels, robust viability kernel (greatest fixed point), feasible empowerment, and packaging endomap/idempotence defect
- **Ring-world environment**: a minimal discrete substrate with toggles for the six primitives (protocol holonomy, repair/accounting, feasibility gating, identity/staging, operator rewriting)
- **Reproducible evidence suite**: audited artifacts, hashed configs, and scripts that regenerate all exhibits and paper assets
- **Lean anchor**: a formal lemma that finite viability iteration stabilizes at the greatest fixed point

## Scope and limitations

The paper is explicit about what it does and does not establish:

- The substrate is intentionally finite and minimal; results are witnesses, not general theorems about all agents
- Empowerment is a difference-making proxy, not a goal or preference theory
- Packaging/objecthood depends on the chosen lens and horizon (explicitly controlled here)
- Null regimes are included to guard against false positives (single-action and schedule-trap baselines)

## Install

```bash
pip install -r requirements.txt
cd lean && lake build
```

## Test

```bash
pytest -q
python scripts/audit_results.py --strict
```

## Run experiments

```bash
python scripts/run_all_experiments.py --clean
python scripts/export_paper_assets.py --no-run
```

## Build paper

```bash
python scripts/build_paper.py
```
