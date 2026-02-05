# Manuscript Consistency Checklist

## 1) Theorem-by-theorem assumption audit

| Label/ID | Type | Assumption tags cited | Assumptions present? | Notes |
|---|---|---|---|---|
| lem:closed-antitone | Lemma | NONE | YES | assumptions inline |
| lem:closure-iterate-stabilizes | Lemma | NONE | YES | assumptions inline |
| cor:closure-saturates | Corollary | NONE | YES | assumptions inline |
| NO_LABEL | Lemma | NONE | YES | assumptions inline |
| NO_LABEL | Theorem | A_FIN, A_LENS | YES | assumptions inline |
| NO_LABEL | Theorem | A_FIN, A_LENS | YES | assumptions inline |
| NO_LABEL | Lemma | A_FIN, A_REV | YES | assumptions inline |
| NO_LABEL | Theorem | A_FIN, A_REV | YES | assumptions inline |
| NO_LABEL | Corollary | A_FIN, A_REV, A_NULL | YES | assumptions inline |
| thm:dpi_path | Theorem | A_FIN, A_LENS | YES | assumptions inline |
| NO_LABEL | Corollary | A_FIN, A_LENS | YES | assumptions inline |
| thm:protocol_trap | Theorem | A_AUT, A_FIN | YES | assumptions inline |
| thm:meta-prim | Theorem | D-META-PROC-01, D-META-LENS-01, D-META-REF-01, D-META-BND-01 | YES | META hypotheses explicit in statement |
| NO_LABEL | Corollary | A_FIN, A_REV, A_AUT, A_ACC, A_NULL | YES | assumptions inline |
| lem:count-definable | Lemma | A_FIN, A_LENS | YES | assumptions inline |
| thm:finite-forcing | Theorem | A_FIN, A_LENS | YES | assumptions inline |
| cor:strict-language-extension | Corollary | A_FIN, A_LENS | YES | assumptions inline |
| NO_LABEL | Theorem | A_FIN, A_REV | YES | assumptions inline |
| NO_LABEL | Theorem | A_FIN, A_REV | YES | assumptions inline |
| lem:work-quantum | Lemma | NONE | YES | local hypotheses inline |
| lem:latency-lb | Lemma | NONE | YES | local hypotheses inline |
| thm:no-zeno | Theorem | NONE | YES | local hypotheses inline |
| lem:tk-route-capacity | Lemma | NONE | YES | assumptions inline |
| lem:tk-causal-compose | Lemma | NONE | YES | assumptions inline |
| lem:tk-parallel-add | Lemma | NONE | YES | assumptions inline |

## 2) Override-sensitive phrase grep list

### Patterns + rationale
- `P3 alone` — Guard against P3-only arrow claims
- `protocol alone` — Guard against protocol-alone arrow claims
- `holonomy alone` — Guard against holonomy-only arrow claims
- `protocol.*yields.*arrow` — Guard against protocol implies arrow claims
- `stroboscopic.*irreversib` — Guard against stroboscopic irreversibility claims
- `external schedule` — Ensure only in clock-audit context
- `layer birth` — Review remaining legacy terminology
- `coarse[- ]graining can create` — DPI non-creation
- `coarse[- ]graining creates` — DPI non-creation
- `macro.*irreversib.*create` — DPI non-creation
- `forgetting variables.*creates` — DPI non-creation
- `data processing` — Should be contraction direction
- `E_{\\tau,\\Pi}.*closure operator` — Avoid empirical/order-closure conflation
- `ClosureOperator.*E_{\\tau,\\Pi}` — Avoid empirical/order-closure conflation
- `E_{\\tau,\\Pi}.*extensive` — Avoid closure-operator axioms

### Hits and review (manuscript/paper.tex)
- Pattern: `P3 alone`
- NO_HITS
- Pattern: `protocol alone`
- NO_HITS
- Pattern: `holonomy alone`
- 714:Under \textbf{A\_AUT}+\textbf{A\_ACC}+\textbf{A\_NULL} (no affinities anywhere in the lifted state space), protocol holonomy alone does not yield sustained arrow-of-time in the autonomous accounted model. -> OK (negated)
- Pattern: `protocol.*yields.*arrow`
- NO_HITS
- Pattern: `stroboscopic.*irreversib`
- NO_HITS
- Pattern: `external schedule`
- 164:  \item[\textbf{A\_AUT}] Autonomy: time-homogeneous dynamics; any protocol/phase variables are included in the state (no external schedule in main-text statements). -> OK (scoped)
- 709:(a) an external schedule (violating \textbf{A\_AUT}), or -> OK (scoped)
- 715:To obtain nonzero steady-state entropy production, one needs a nontrivial affinity component (P6) in the lifted dynamics (e.g.\ a biased phase cycle) or an external schedule. -> OK (scoped)
- 831:empirical endomap $E_{\tau,f}$. This is a change in the operator itself, not an external schedule. -> OK (scoped)
- 843:No external schedule is assumed in the main text; externally scheduled stroboscopic protocols are -> OK (scoped)
- 932:\subsection{Protocol trap: external schedule vs autonomous lifted model} -> OK (scoped)
- Pattern: `coarse[- ]graining can create`
- NO_HITS
- Pattern: `coarse[- ]graining creates`
- NO_HITS
- Pattern: `macro.*irreversib.*create`
- NO_HITS
- Pattern: `forgetting variables.*creates`
- NO_HITS
- Pattern: `data processing`
- 56:Coarse-graining cannot create arrow-of-time: the path reversal asymmetry contracts under pushforward (data processing in Section~\ref{sec:aot}). -> OK (scoped)
- 100:\begin{remark}[Stationarity is not needed for $\Sigma_T$ or data processing] -> OK (scoped)
- 102:The data processing inequality for KL is purely measure-theoretic and therefore applies to $\Sigma_T(\rho)$ without any stationarity assumption. -> OK (scoped)
- 139:Finally, Sections~\ref{sec:acc} and \ref{sec:aot} provide two audit families used here: graph 1-form exactness/affinities (ACC) and path reversal asymmetry (arrow-of-time), the latter enjoying a precise monotonicity principle under coarse-graining (data processing). -> OK (scoped)
- 627:\section{Arrow-of-time as path reversal asymmetry; data processing; protocol trap}\label{sec:aot} -> OK (scoped)
- 1071:\paragraph{Path reversal asymmetry and data processing (DPI).} -> OK (scoped)
- Pattern: `E_{\\tau,\\Pi}.*closure operator`
- NO_HITS
- Pattern: `ClosureOperator.*E_{\\tau,\\Pi}`
- NO_HITS
- Pattern: `E_{\\tau,\\Pi}.*extensive`
- NO_HITS

## 3) Checks run + results

- python3 scripts/check_paper_contract.py: PASS
- python3 scripts/check_deps_dag.py: PASS
- python3 scripts/check_kb_pointers.py: SKIPPED (no knowledge-base refs in manuscript)
- python3 scripts/check_tex_refs.py: PASS
- ./check_lean.sh: PASS
- cd python && .venv/bin/python -m pytest -q: PASS
