# TechRxiv Submission Notes

Copy/paste-ready metadata for the TechRxiv upload form.

---

## Title

To Throw a Stone with Six Birds: On Agents and Agenthood

## Abstract

What does it take for a system to have choices that make a difference? We study this question in finite controlled Markov kernels, without assuming goals, utilities, or an inner agent. Following Six Birds Theory (SBT), we treat an agent as a theory object: a package that is maintained inside an induced description of the system and that has a budgeted interface to the rest of the world. This lets us separate agenthood, the existence and persistence of such a package, from agency, the ability of its interface choices to change what happens outside. Three computable measures make the distinction concrete: the viability kernel (the largest set of safe states in which every state has an affordable action whose possible successors all remain in the set), feasible empowerment (the channel capacity from affordable action sequences to a later outside variable), and the idempotence defect of a coarse-grained packaging map. Two null tests guard against false positives: a single-action system has zero empowerment, and an external schedule mistaken for an action yields a spurious 1 bit. In a small ring world with switchable mechanisms we then obtain three controlled comparisons. With funded preventive repair, the packaging defect at a full phase cycle falls from 1 to 0 and the set of states that can be kept coherent grows from empty to 16; controls show that a zero defect alone does not identify maintenance. A phase-dependent step leaves one-step empowerment unchanged but raises the sampled median at two steps from 1.21 to 1.72 bits. Reducing slip raises median empowerment from 0.84 to 1.37 bits. A Lean 4 development proves that the viability iteration reaches the greatest safe controlled-invariant set after at most as many strict removals as there are safe states. All results are finite witnesses produced by deterministic, audited scripts.

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
- **Zenodo DOI (code):** https://doi.org/10.5281/zenodo.18451887
- **GitHub repository:** https://github.com/ioannist/six-birds-agent
- **SBT Foundations reference:** https://doi.org/10.5281/zenodo.18365949

## License Recommendation

**Recommended: CC BY 4.0**

**Rationale:** CC BY 4.0 (Creative Commons Attribution) is the most widely adopted open-access license for preprints. It maximizes redistribution and reuse while requiring citation, which is standard academic practice. TechRxiv supports this license. If the author later publishes in a journal that requires exclusive rights transfer, note that a CC BY preprint version remains permanently available under that license (which is the standard expectation for preprints). If the author prefers to retain maximum flexibility for future publisher negotiations, "No license" is an alternative---but this limits reuse and may reduce citation and discoverability.

The paper already carries CC-BY 4.0 in its footer, so this is consistent.
