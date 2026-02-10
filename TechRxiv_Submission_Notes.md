# TechRxiv Submission Notes

Copy/paste-ready metadata for the TechRxiv upload form.

---

## Title

To Throw a Stone with Six Birds: On Agents and Agenthood

## Abstract

We present a reproducible computational framework for defining and measuring agency in finite controlled Markov kernels without assuming goals, utilities, or homunculi. The framework operationalizes three complementary metrics: (i) viability kernels (greatest controlled-invariant safe sets under ledger-gated feasibility and successor-support semantics), (ii) feasible empowerment (channel capacity of the induced action-to-output channel restricted by budgets), and (iii) an empirical packaging endomap whose idempotence defect quantifies objecthood for coarse lenses. Building on Six Birds Theory (SBT), we propose that an agent is a theory object---a maintained package inside an induced layer with a ledger-gated interface---and we separate agenthood (enablement: the layer exists and persists) from agency (causation: interface interventions change outside futures). Across matched-control ablations in a minimal ring-world substrate, we obtain four auditable separations: (1) calibrated null regimes prevent false positives (single-action: 0-bit empowerment; schedule trap: 1 bit under the wrong model, 0 under the correct one); (2) enabling repair collapses idempotence defect at tau=2 from 1 to 0; (3) enabling protocol holonomy leaves H=1 unchanged but increases empowerment for H>=2; and (4) operator rewriting (skill theta) monotonically increases median empowerment (0.73 to 1.34 bits). All experiments are driven by deterministic, hash-traceable scripts with a strict audit mode, and a Lean 4 anchor formalizes the viability-kernel fixed-point property. We are explicit that stronger claims about goals, consciousness, or real organisms are not established here; our conclusions rely on finite witnesses and operational proxies under explicit controls.

## Keywords

- agency
- agenthood
- viability kernel
- controlled invariance
- empowerment
- channel capacity
- Markov decision processes
- reproducible artifacts
- formal methods
- causal control

## Author

- **Ioannis Tsiokos**
  - Affiliation: Automorph Inc., Wilmington, DE, USA
  - Email: ioannis@automorph.io
  - ORCID: 0009-0009-7659-5964

## Suggested TechRxiv Categories / Subject Tags

**Primary:**
- Computer Science --- Algorithms and Theory

**Secondary (choose 1--2 as applicable):**
- Computer Science --- Artificial Intelligence
- Computer Science --- Software Engineering

**Justification:** The paper presents a computational framework for computing viability kernels, channel capacity (empowerment), and packaging diagnostics on finite controlled Markov kernels, with deterministic reproducible scripts, a strict artifact auditor, and a mechanized Lean proof. The core contributions are algorithmic methods and software artifacts, not pure mathematics or philosophy.

## Links

- **Zenodo DOI (paper):** https://doi.org/10.5281/zenodo.18439737
- **GitHub repository:** https://github.com/ioannist/six-birds-agent
- **SBT Foundations reference:** https://doi.org/10.5281/zenodo.18365949

## License Recommendation

**Recommended: CC BY 4.0**

**Rationale:** CC BY 4.0 (Creative Commons Attribution) is the most widely adopted open-access license for preprints. It maximizes redistribution and reuse while requiring citation, which is standard academic practice. TechRxiv supports this license. If the author later publishes in a journal that requires exclusive rights transfer, note that a CC BY preprint version remains permanently available under that license (which is the standard expectation for preprints). If the author prefers to retain maximum flexibility for future publisher negotiations, "No license" is an alternative---but this limits reuse and may reduce citation and discoverability.

The paper already carries CC-BY 4.0 in its footer, so this is consistent.
