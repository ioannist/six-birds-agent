import Mathlib

namespace Agency

open Finset

variable {α : Type} [DecidableEq α] [Fintype α]

def K (F : Finset α → Finset α) (n : Nat) : Finset α :=
  Nat.iterate F n Finset.univ

lemma K_succ_subset (F : Finset α → Finset α) (hsub : ∀ s, F s ⊆ s) (n : Nat) :
    K F (n + 1) ⊆ K F n := by
  simpa [K, Function.iterate_succ_apply'] using (hsub (K F n))

lemma K_card_le (F : Finset α → Finset α) (hsub : ∀ s, F s ⊆ s) (n : Nat) :
    (K F (n + 1)).card ≤ (K F n).card := by
  simpa using (Finset.card_le_card (K_succ_subset F hsub n))


theorem iterate_top_greatest_fixpoint
    (F : Finset α → Finset α) (hmono : Monotone F) (hsub : ∀ s, F s ⊆ s) :
    ∃ n : Nat,
      let K := Nat.iterate F n Finset.univ
      F K = K ∧ ∀ S : Finset α, F S = S → S ⊆ K := by
  classical
  let Kfn : Nat → Finset α := fun n => K F n

  obtain ⟨n, hmin⟩ :=
    exists_minimalFor_of_wellFoundedLT
      (α := Nat)
      (P := fun _ : Nat => True)
      (f := fun n => (Kfn n).card)
      ⟨0, trivial⟩

  have hmin_le : ∀ m, (Kfn n).card ≤ (Kfn m).card := by
    intro m
    by_cases h : (Kfn m).card ≤ (Kfn n).card
    · exact hmin.2 (by trivial) h
    · exact le_of_lt (lt_of_not_ge h)

  have h_eq : Kfn (n + 1) = Kfn n := by
    apply Finset.eq_of_subset_of_card_le (K_succ_subset F hsub n)
    exact hmin_le (n + 1)

  have h_fix : F (Kfn n) = Kfn n := by
    have hsucc : F (Kfn n) = Kfn (n + 1) := by
      simpa [Kfn, K, Function.iterate_succ_apply']
    calc
      F (Kfn n) = Kfn (n + 1) := hsucc
      _ = Kfn n := h_eq

  have h_greatest : ∀ S : Finset α, F S = S → S ⊆ Kfn n := by
    intro S hFS
    have h_subset_all : ∀ k, S ⊆ Kfn k := by
      intro k
      induction k with
      | zero =>
          simpa [Kfn, K] using (Finset.subset_univ S)
      | succ k ih =>
          have hmono' : F S ⊆ F (Kfn k) := hmono ih
          simpa [hFS, Kfn, K, Function.iterate_succ_apply'] using hmono'
    exact h_subset_all n

  refine ⟨n, ?_⟩
  simpa [Kfn, K] using And.intro h_fix h_greatest

end Agency
