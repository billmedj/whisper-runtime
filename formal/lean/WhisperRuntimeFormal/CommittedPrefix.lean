import Std

namespace WhisperRuntimeFormal.CommittedPrefix

/-- One timestamped output window. -/
structure Window where
  startMs : Nat
  endMs : Nat
deriving DecidableEq, Repr

/-- Use an explicit claim when present; otherwise preserve the prior prefix. -/
def nextPrefix (before claim : Option Nat) : Option Nat :=
  match claim with
  | some value => some value
  | none => before

/--
The guards applied before a result can enter the session log.

An existing committed prefix cannot be overlapped or moved backward. A new
claim cannot extend beyond the result that carries it.
-/
def admissible (before : Option Nat) (window : Window) (claim : Option Nat) : Prop :=
  window.startMs ≤ window.endMs ∧
    (match before with
      | some value => value ≤ window.startMs
      | none => True) ∧
    (match before, claim with
      | some old, some new => old ≤ new
      | _, _ => True) ∧
    (match claim with
      | some value => value ≤ window.endMs
      | none => True)

/-- Decide the guard after reducing the two optional inputs. -/
private def admissibleDecidable
    (before : Option Nat)
    (window : Window)
    (claim : Option Nat) : Decidable (admissible before window claim) := by
  unfold admissible
  cases before <;> cases claim <;> infer_instance

/-- Apply one committed-prefix transition, or reject the result. -/
def append
    (before : Option Nat)
    (window : Window)
    (claim : Option Nat) : Option (Option Nat) := by
  letI := admissibleDecidable before window claim
  exact
    if admissible before window claim then
      some (nextPrefix before claim)
    else
      none

theorem accepted_append_respects_committed_prefix
    {frontier : Nat}
    {window : Window}
    {claim after : Option Nat}
    (accepted : append (some frontier) window claim = some after) :
    frontier ≤ window.startMs := by
  by_cases allowed : admissible (some frontier) window claim
  · exact allowed.2.1
  · simp [append, allowed] at accepted

theorem accepted_append_does_not_regress
    {old new : Nat}
    {window : Window}
    {claim : Option Nat}
    (accepted : append (some old) window claim = some (some new)) :
    old ≤ new := by
  by_cases allowed : admissible (some old) window claim
  · cases claim with
    | none =>
        have same : some old = some new := by
          simpa [append, allowed, nextPrefix] using accepted
        exact Nat.le_of_eq (Option.some.inj same)
    | some claimed =>
        have same : some claimed = some new := by
          simpa [append, allowed, nextPrefix] using accepted
        have equal : claimed = new := Option.some.inj same
        simpa [equal] using allowed.2.2.1
  · simp [append, allowed] at accepted

theorem accepted_claim_is_bounded_by_window
    {claimed : Nat}
    {before after : Option Nat}
    {window : Window}
    (accepted : append before window (some claimed) = some after) :
    claimed ≤ window.endMs ∧ after = some claimed := by
  by_cases allowed : admissible before window (some claimed)
  · constructor
    · exact allowed.2.2.2
    · simpa [append, allowed, nextPrefix] using accepted.symm
  · simp [append, allowed] at accepted

theorem accepted_append_without_claim_preserves_prefix
    {before after : Option Nat}
    {window : Window}
    (accepted : append before window none = some after) :
    after = before := by
  by_cases allowed : admissible before window none
  · simpa [append, allowed, nextPrefix] using accepted.symm
  · simp [append, allowed] at accepted

end WhisperRuntimeFormal.CommittedPrefix
