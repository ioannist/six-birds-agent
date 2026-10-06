import Mathlib.Data.Fintype.Card
import Mathlib.Logic.Function.Iterate
import Lean.Elab.Tactic.Omega

namespace Agency

variable {α : Type}

def K [Fintype α] (F : Finset α → Finset α) (n : Nat) : Finset α :=
  Nat.iterate F n Finset.univ

lemma K_succ_subset [Fintype α] (F : Finset α → Finset α) (hsub : ∀ s, F s ⊆ s) (n : Nat) :
    K F (n + 1) ⊆ K F n := by
  simpa [K, Function.iterate_succ_apply'] using (hsub (K F n))

lemma K_card_le [Fintype α] (F : Finset α → Finset α) (hsub : ∀ s, F s ⊆ s) (n : Nat) :
    (K F (n + 1)).card ≤ (K F n).card := by
  simpa using (Finset.card_le_card (K_succ_subset F hsub n))


/-- A contracting finite-set iteration stabilizes after at most `start.card`
strict removals, including an empty initial set. -/
theorem iterate_stabilizes_bounded
    (F : Finset α → Finset α) (hsub : ∀ s, F s ⊆ s) (start : Finset α) :
    ∃ n ≤ start.card, F (Nat.iterate F n start) = Nat.iterate F n start := by
  classical
  by_contra h
  have hlt (n : Nat) (hn : n ≤ start.card) :
      (Nat.iterate F (n + 1) start).card < (Nat.iterate F n start).card := by
    have hs : Nat.iterate F (n + 1) start ⊆ Nat.iterate F n start := by
      simpa [Function.iterate_succ_apply'] using hsub (Nat.iterate F n start)
    have hle := Finset.card_le_card hs
    have hnot : ¬ (Nat.iterate F n start).card ≤
        (Nat.iterate F (n + 1) start).card := by
      intro hc
      have heq := Finset.eq_of_subset_of_card_le hs hc
      apply h
      exact ⟨n, hn, by simpa [Function.iterate_succ_apply'] using heq⟩
    omega
  have hcards : ∀ n, n ≤ start.card + 1 →
      (Nat.iterate F n start).card + n ≤ start.card := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
        intro hn
        have hi := ih (by omega)
        have hd := hlt n (by omega)
        omega
  have hc := hcards (start.card + 1) (by omega)
  omega

/-- The same result from any initial set, with greatestness relative to that
set. Starting at `safe` gives the implementation's iteration exactly. -/
theorem iterate_greatest_fixpoint_bounded_from
    (F : Finset α → Finset α) (hmono : Monotone F)
    (hsub : ∀ s, F s ⊆ s) (start : Finset α) :
    ∃ n ≤ start.card,
      let K := Nat.iterate F n start
      F K = K ∧ ∀ S : Finset α, S ⊆ start → F S = S → S ⊆ K := by
  obtain ⟨n, hn, hfix⟩ := iterate_stabilizes_bounded F hsub start
  refine ⟨n, hn, hfix, ?_⟩
  intro S hS hFS
  have hall : ∀ k, S ⊆ Nat.iterate F k start := by
    intro k
    induction k with
    | zero => simpa using hS
    | succ k ih =>
        simpa [hFS, Function.iterate_succ_apply'] using hmono ih
  exact hall n

variable {β : Type}

/-- Robust viability with explicit safety, admissible actions, and successor
sets. In stochastic instances successor sets must be actual positive support;
this abstraction does not assert that floating kernel code realizes them. -/
noncomputable def viabilityOperator
    (safe : Finset α) (feasible : α → Finset β) (post : α → β → Finset α)
    (K : Finset α) : Finset α := by
  classical
  exact K.filter fun s => s ∈ safe ∧ ∃ a ∈ feasible s, post s a ⊆ K

lemma viabilityOperator_contracting (safe : Finset α) (feasible : α → Finset β)
    (post : α → β → Finset α) (K : Finset α) :
    viabilityOperator safe feasible post K ⊆ K := by
  classical
  exact Finset.filter_subset _ _

lemma viabilityOperator_monotone (safe : Finset α) (feasible : α → Finset β)
    (post : α → β → Finset α) : Monotone (viabilityOperator safe feasible post) := by
  classical
  intro K L hKL s hs
  obtain ⟨hK, hsafe, a, ha, hpost⟩ := Finset.mem_filter.mp hs
  exact Finset.mem_filter.mpr ⟨hKL hK, hsafe, a, ha, fun t ht => hKL (hpost ht)⟩

/-- Fixed points are precisely safe controlled-invariant sets. -/
theorem viabilityOperator_fixpoint_iff (safe : Finset α) (feasible : α → Finset β)
    (post : α → β → Finset α) (K : Finset α) :
    viabilityOperator safe feasible post K = K ↔
      K ⊆ safe ∧ ∀ s ∈ K, ∃ a ∈ feasible s, post s a ⊆ K := by
  classical
  constructor
  · intro h
    have hm (s : α) (hs : s ∈ K) :
        s ∈ safe ∧ ∃ a ∈ feasible s, post s a ⊆ K := by
      have hh : s ∈ viabilityOperator safe feasible post K := h.symm ▸ hs
      exact (Finset.mem_filter.mp hh).2
    exact ⟨fun s hs => (hm s hs).1, fun s hs => (hm s hs).2⟩
  · rintro ⟨hsafe, hinv⟩
    apply Finset.Subset.antisymm (viabilityOperator_contracting safe feasible post K)
    intro s hs
    exact Finset.mem_filter.mpr ⟨hs, hsafe hs, hinv s hs⟩

/-- A direct specialization to robust viability, including the safe-set bound
and maximality among all safe controlled-invariant sets. -/
theorem viability_iteration_greatest_bounded (safe : Finset α)
    (feasible : α → Finset β) (post : α → β → Finset α) :
    ∃ n ≤ safe.card,
      let K := Nat.iterate (viabilityOperator safe feasible post) n safe
      (K ⊆ safe ∧ ∀ s ∈ K, ∃ a ∈ feasible s, post s a ⊆ K) ∧
      ∀ S : Finset α, S ⊆ safe →
        (∀ s ∈ S, ∃ a ∈ feasible s, post s a ⊆ S) → S ⊆ K := by
  obtain ⟨n, hn, hfix, hgreatest⟩ := iterate_greatest_fixpoint_bounded_from
    (viabilityOperator safe feasible post) (viabilityOperator_monotone safe feasible post)
    (viabilityOperator_contracting safe feasible post) safe
  refine ⟨n, hn, (viabilityOperator_fixpoint_iff safe feasible post _).mp hfix, ?_⟩
  intro S hsafe hinv
  exact hgreatest S hsafe ((viabilityOperator_fixpoint_iff safe feasible post S).mpr ⟨hsafe, hinv⟩)

/-- The original top-set theorem, strengthened with a cardinal bound. -/
theorem iterate_top_greatest_fixpoint_bounded [Fintype α]
    (F : Finset α → Finset α) (hmono : Monotone F) (hsub : ∀ s, F s ⊆ s) :
    ∃ n ≤ Fintype.card α,
      let K := Nat.iterate F n Finset.univ
      F K = K ∧ ∀ S : Finset α, F S = S → S ⊆ K := by
  obtain ⟨n, hn, hfix, hgreatest⟩ :=
    iterate_greatest_fixpoint_bounded_from F hmono hsub Finset.univ
  exact ⟨n, by simpa using hn, hfix, fun S hS => hgreatest S (Finset.subset_univ S) hS⟩

/-- Backward-compatible exported anchor from the manuscript. -/
theorem iterate_top_greatest_fixpoint [Fintype α]
    (F : Finset α → Finset α) (hmono : Monotone F) (hsub : ∀ s, F s ⊆ s) :
    ∃ n : Nat,
      let K := Nat.iterate F n Finset.univ
      F K = K ∧ ∀ S : Finset α, F S = S → S ⊆ K := by
  obtain ⟨n, _, h⟩ := iterate_top_greatest_fixpoint_bounded F hmono hsub
  exact ⟨n, h⟩

/-- Controlled invariance supplies a single stationary feedback selector on K.
No action or inhabitant is assumed outside K, which can be empty. -/
theorem controlledInvariant_stationary_policy (K : Finset α)
    (feasible : α → Finset β) (post : α → β → Finset α)
    (hinv : ∀ s ∈ K, ∃ a ∈ feasible s, post s a ⊆ K) :
    ∃ policy : {s // s ∈ K} → β,
      ∀ s, policy s ∈ feasible s.val ∧ post s.val (policy s) ⊆ K := by
  classical
  refine ⟨fun s => Classical.choose (hinv s.val s.property), ?_⟩
  intro s
  exact Classical.choose_spec (hinv s.val s.property)

end Agency

namespace Agency

/-- Support invariance propagates along every trajectory of the selector.
This is pathwise safety; it does not assume probabilities or expected safety. -/
theorem stationary_policy_paths_safe {α β : Type} (K : Finset α)
    (post : α → β → Finset α) (policy : α → β)
    (hinv : ∀ s ∈ K, post s (policy s) ⊆ K)
    (path : Nat → α) (hstart : path 0 ∈ K)
    (hstep : ∀ n, path (n + 1) ∈ post (path n) (policy (path n))) :
    ∀ n, path n ∈ K := by
  intro n
  induction n with
  | zero => exact hstart
  | succ n ih => exact hinv (path n) ih (hstep n)

end Agency
