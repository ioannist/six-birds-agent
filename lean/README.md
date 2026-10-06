# Finite robust viability

From this directory, run:

```bash
lake build
lake env lean Audit.lean
```

The pinned toolchain and mathlib revision are v4.27.0. If dependencies lack
compiled objects, fetch the relevant cache before building:

```bash
lake exe cache get Mathlib.Data.Fintype.Card Mathlib.Logic.Function.Iterate
```

`Agency/Viability.lean` proves finite-set stabilization within the starting
set's cardinality, greatestness among fixed points below that set, and a direct
specialization to the robust viability operator. Its monotonicity and
contraction are proved from safety, feasible-action sets, and successor sets.
It also supplies a stationary feedback selector on the viable subtype and
proves safety along every support-respecting trajectory of an invariant policy.
The original `iterate_top_greatest_fixpoint` export remains available.

The arbitrary-start theorems need only a finite starting set; the ambient state
type need not be finite. Stochastic interpretations require nonempty successor
support. The general set operator itself permits empty successor sets.

`Audit.lean` prints transitive axiom dependencies. The proofs use only Lean's
standard `propext`, `Classical.choice`, and `Quot.sound`; there are no proof
holes, custom axioms, or `native_decide` calls.

The Python implementation, ring probabilities, packaging endomaps, and channel
capacities are not mechanized here. Their mathematical checks and the limits
of the bridge to these theorems are recorded in
[`../docs/mathematics-review.txt`](../docs/mathematics-review.txt).
