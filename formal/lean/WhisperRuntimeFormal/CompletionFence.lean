import Std

namespace WhisperRuntimeFormal.CompletionFence

/--
The phases of one admitted inference transaction.

`ready` means that the backend completion fence established quiescence.  The
result is still private in that phase.  `recovered` means that an external
recovery procedure established that retained resources can be released; it
does not publish a result.
-/
inductive Phase where
  | executing
  | ready
  | quarantined
  | committed
  | discarded
  | recovered
deriving DecidableEq, Repr

/-- The only two observations made at the completion boundary. -/
inductive FenceObservation where
  | quiescent
  | uncertain
deriving DecidableEq, Repr

/-- Whether the transaction still accounts for its admitted capacity. -/
inductive CapacityDisposition where
  | held
  | released
deriving DecidableEq, Repr

/-- Whether the transaction result is externally visible. -/
inductive PublicationDisposition where
  | withheld
  | published
deriving DecidableEq, Repr

/-- The evidence class that permits capacity release. -/
inductive ReleaseBasis where
  | completionFence
  | recovery
deriving DecidableEq, Repr

def capacityDisposition : Phase → CapacityDisposition
  | .executing => .held
  | .ready => .held
  | .quarantined => .held
  | .committed => .released
  | .discarded => .released
  | .recovered => .released

def publicationDisposition : Phase → PublicationDisposition
  | .committed => .published
  | _ => .withheld

def releaseBasis : Phase → Option ReleaseBasis
  | .committed => some .completionFence
  | .discarded => some .completionFence
  | .recovered => some .recovery
  | _ => none

/-- Observe the backend fence once, while execution is in progress. -/
def observeFence : Phase → FenceObservation → Option Phase
  | .executing, .quiescent => some .ready
  | .executing, .uncertain => some .quarantined
  | _, _ => none

/-- Publish only a private result whose completion fence established quiescence. -/
def publish : Phase → Option Phase
  | .ready => some .committed
  | _ => none

/-- Discard a private result after its completion fence established quiescence. -/
def discard : Phase → Option Phase
  | .ready => some .discarded
  | _ => none

/--
Finish a recovery procedure for a quarantined transaction.

This transition represents successful recovery evidence supplied by the
implementation.  This model does not prove that a CUDA driver or process
actually produced that evidence.
-/
def recover : Phase → Option Phase
  | .quarantined => some .recovered
  | _ => none

/-- The complete transition relation for this small state machine. -/
inductive Step : Phase → Phase → Prop where
  | fenceCompleted : Step .executing .ready
  | fenceUncertain : Step .executing .quarantined
  | publish : Step .ready .committed
  | discard : Step .ready .discarded
  | recover : Step .quarantined .recovered

/-- States reachable from a newly executing, already admitted transaction. -/
inductive Reachable : Phase → Prop where
  | initial : Reachable .executing
  | next {before after : Phase} : Reachable before → Step before after → Reachable after

theorem uncertain_observation_enters_quarantine :
    observeFence .executing .uncertain = some .quarantined := by
  rfl

theorem quiescent_observation_makes_result_ready :
    observeFence .executing .quiescent = some .ready := by
  rfl

theorem accepted_publication_requires_ready
    {before after : Phase}
    (accepted : publish before = some after) :
    before = .ready ∧ after = .committed := by
  cases before <;> simp [publish] at accepted
  exact ⟨rfl, accepted.symm⟩

theorem quarantined_result_is_withheld :
    publicationDisposition .quarantined = .withheld := by
  rfl

theorem quarantined_capacity_is_held :
    capacityDisposition .quarantined = .held := by
  rfl

theorem quarantined_cannot_publish :
    publish .quarantined = none := by
  rfl

theorem capacity_released_iff (phase : Phase) :
    capacityDisposition phase = .released ↔
      phase = .committed ∨ phase = .discarded ∨ phase = .recovered := by
  cases phase <;> simp [capacityDisposition]

theorem published_result_has_completed_fence_basis
    (phase : Phase)
    (published : publicationDisposition phase = .published) :
    releaseBasis phase = some .completionFence := by
  cases phase <;> simp [publicationDisposition, releaseBasis] at published ⊢

theorem released_capacity_has_completion_or_recovery_basis
    (phase : Phase)
    (released : capacityDisposition phase = .released) :
    releaseBasis phase = some .completionFence ∨
      releaseBasis phase = some .recovery := by
  cases phase <;>
    simp [capacityDisposition, releaseBasis] at released ⊢

theorem only_recovery_exits_quarantine
    {after : Phase}
    (step : Step .quarantined after) :
    after = .recovered := by
  cases step
  rfl

theorem terminal_states_have_no_successor
    {before after : Phase}
    (terminal : before = .committed ∨ before = .discarded ∨ before = .recovered) :
    ¬Step before after := by
  intro step
  rcases terminal with rfl | rfl | rfl <;> cases step

theorem reachable_ready_has_completed_fence
    (reachable : Reachable .ready) :
    Step .executing .ready := by
  cases reachable with
  | next _ step =>
      cases step
      exact Step.fenceCompleted

theorem reachable_quarantine_has_uncertain_fence
    (reachable : Reachable .quarantined) :
    Step .executing .quarantined := by
  cases reachable with
  | next _ step =>
      cases step
      exact Step.fenceUncertain

theorem reachable_commit_has_fenced_publication
    (reachable : Reachable .committed) :
    Step .executing .ready ∧ Step .ready .committed := by
  cases reachable with
  | next readyReachable step =>
      cases step
      exact ⟨reachable_ready_has_completed_fence readyReachable, Step.publish⟩

theorem reachable_recovery_has_quarantine_sequence
    (reachable : Reachable .recovered) :
    Step .executing .quarantined ∧ Step .quarantined .recovered := by
  cases reachable with
  | next quarantinedReachable step =>
      cases step
      exact ⟨reachable_quarantine_has_uncertain_fence quarantinedReachable, Step.recover⟩

theorem reachable_release_has_completion_or_recovery_basis
    {phase : Phase}
    (_reachable : Reachable phase)
    (released : capacityDisposition phase = .released) :
    releaseBasis phase = some .completionFence ∨
      releaseBasis phase = some .recovery := by
  exact released_capacity_has_completion_or_recovery_basis phase released

end WhisperRuntimeFormal.CompletionFence
