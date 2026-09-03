import Std

namespace WhisperRuntimeFormal

abbrev RuntimeId := Nat
abbrev LeaseId := Nat
abbrev SessionId := Nat
abbrev Revision := Nat
abbrev Payload := Nat

/-- A two-dimensional resource quantity. -/
structure Budget where
  memory : Nat
  compute : Nat
deriving DecidableEq, Repr

namespace Budget

def Fits (need available : Budget) : Prop :=
  need.memory ≤ available.memory ∧ need.compute ≤ available.compute

instance fitsDecidable (need available : Budget) : Decidable (Fits need available) := by
  unfold Fits
  infer_instance

def add (left right : Budget) : Budget :=
  { memory := left.memory + right.memory
    compute := left.compute + right.compute }

theorem add_assoc (first second third : Budget) :
    add (add first second) third = add first (add second third) := by
  cases first
  cases second
  cases third
  simp [add, Nat.add_assoc]

def reserve (capacity held : Budget) : Budget :=
  { memory := capacity.memory - held.memory
    compute := capacity.compute - held.compute }

theorem reserve_fits_capacity (capacity held : Budget) :
    Fits (reserve capacity held) capacity := by
  constructor <;> simp [reserve]

theorem fits_trans {left middle right : Budget}
    (first : Fits left middle)
    (second : Fits middle right) : Fits left right := by
  exact ⟨Nat.le_trans first.1 second.1, Nat.le_trans first.2 second.2⟩

end Budget

inductive LeaseStatus where
  | active
  | committed
  | aborted
deriving DecidableEq, Repr

/-- A lease records its issuer, allocation, and observed session revision. -/
structure Lease where
  id : LeaseId
  owner : RuntimeId
  session : SessionId
  baseVersion : Revision
  reserved : Budget
  payload : Payload
deriving DecidableEq, Repr

/-- Shared state. Only active leases consume capacity. Terminal IDs remain recorded. -/
structure Runtime where
  id : RuntimeId
  capacity : Budget
  version : SessionId → Revision
  active : List Lease
  committed : List LeaseId
  aborted : List LeaseId

def setVersion
    (versions : SessionId → Revision)
    (session : SessionId)
    (revision : Revision) : SessionId → Revision :=
  fun current => if current = session then revision else versions current

def reservedTotal : List Lease → Budget
  | [] => { memory := 0, compute := 0 }
  | lease :: rest => Budget.add lease.reserved (reservedTotal rest)

/-- Available capacity is derived from the active ledger, not updated independently. -/
def Runtime.available (runtime : Runtime) : Budget :=
  Budget.reserve runtime.capacity (reservedTotal runtime.active)

/-- The active ledger is fully backed by total capacity. -/
def Runtime.LedgerFits (runtime : Runtime) : Prop :=
  Budget.Fits (reservedTotal runtime.active) runtime.capacity

def findLease : List Lease → LeaseId → Option Lease
  | [], _ => none
  | lease :: rest, id =>
      if lease.id = id then some lease else findLease rest id

def eraseLease : List Lease → LeaseId → List Lease
  | [], _ => []
  | lease :: rest, id =>
      if lease.id = id then rest else lease :: eraseLease rest id

def activeIds (leases : List Lease) : List LeaseId :=
  leases.map Lease.id

theorem findLease_none_iff_id_not_active (leases : List Lease) (id : LeaseId) :
    findLease leases id = none ↔ id ∉ activeIds leases := by
  induction leases with
  | nil => simp [findLease, activeIds]
  | cons lease rest inductionHypothesis =>
      by_cases same : lease.id = id
      · subst id
        simp [findLease, activeIds]
      · simp [findLease, activeIds, same, inductionHypothesis, Ne.symm same]

theorem findLease_some_implies_mem
    (leases : List Lease)
    (id : LeaseId)
    (lease : Lease)
    (found : findLease leases id = some lease) :
    lease ∈ leases := by
  induction leases with
  | nil => simp [findLease] at found
  | cons head rest inductionHypothesis =>
      by_cases same : head.id = id
      · simp [findLease, same] at found
        subst lease
        simp
      · simp [findLease, same] at found
        exact List.mem_cons_of_mem head (inductionHypothesis found)

theorem findLease_some_implies_id
    (leases : List Lease)
    (id : LeaseId)
    (lease : Lease)
    (found : findLease leases id = some lease) :
    lease.id = id := by
  induction leases with
  | nil => simp [findLease] at found
  | cons head rest inductionHypothesis =>
      by_cases same : head.id = id
      · simp [findLease, same] at found
        subst lease
        exact same
      · simp [findLease, same] at found
        exact inductionHypothesis found

theorem activeIds_eraseLease (leases : List Lease) (id : LeaseId) :
    activeIds (eraseLease leases id) = (activeIds leases).erase id := by
  induction leases with
  | nil => simp [eraseLease, activeIds]
  | cons lease rest inductionHypothesis =>
      by_cases same : lease.id = id
      · simp [eraseLease, activeIds, same]
      · simp only [eraseLease, same, ↓reduceIte, activeIds, List.map_cons]
        simp [same]
        simpa [activeIds] using inductionHypothesis

theorem eraseLease_subset (leases : List Lease) (id : LeaseId) :
    ∀ {lease}, lease ∈ eraseLease leases id → lease ∈ leases := by
  induction leases with
  | nil => simp [eraseLease]
  | cons head rest inductionHypothesis =>
      intro lease member
      by_cases same : head.id = id
      · simp [eraseLease, same] at member
        exact List.mem_cons_of_mem head member
      · simp [eraseLease, same] at member
        rcases member with rfl | member
        · simp
        · exact List.mem_cons_of_mem head (inductionHypothesis member)

theorem eraseLease_preserves_absence
    (leases : List Lease)
    (erased sought : LeaseId)
    (absent : findLease leases sought = none) :
    findLease (eraseLease leases erased) sought = none := by
  induction leases with
  | nil => simp [eraseLease, findLease]
  | cons head rest inductionHypothesis =>
      by_cases removeHead : head.id = erased
      · have tailAbsent : findLease rest sought = none := by
          by_cases same : head.id = sought
          · simp [findLease, same] at absent
          · simpa [findLease, same] using absent
        simpa [eraseLease, removeHead] using tailAbsent
      · have headDifferent : head.id ≠ sought := by
          intro same
          simp [findLease, same] at absent
        have tailAbsent : findLease rest sought = none := by
          simpa [findLease, headDifferent] using absent
        simp [eraseLease, removeHead, findLease, headDifferent,
          inductionHypothesis tailAbsent]

theorem eraseLease_removes_id_when_unique
    (leases : List Lease)
    (id : LeaseId)
    (unique : (activeIds leases).Nodup) :
    findLease (eraseLease leases id) id = none := by
  rw [findLease_none_iff_id_not_active, activeIds_eraseLease]
  exact fun member => unique.not_mem_erase member

theorem reservedTotal_eraseLease_add_found
    (leases : List Lease)
    (id : LeaseId)
    (lease : Lease)
    (found : findLease leases id = some lease)
    (unique : (activeIds leases).Nodup) :
    Budget.add (reservedTotal (eraseLease leases id)) lease.reserved =
      reservedTotal leases := by
  induction leases with
  | nil => simp [findLease] at found
  | cons head rest inductionHypothesis =>
      by_cases same : head.id = id
      · simp [findLease, same] at found
        subst lease
        simp [eraseLease, same, reservedTotal, Budget.add]
        omega
      · have tailFound : findLease rest id = some lease := by
          simpa [findLease, same] using found
        have tailUnique : (activeIds rest).Nodup := by
          simpa [activeIds] using unique.tail
        have total := inductionHypothesis tailFound tailUnique
        simp only [eraseLease, same, ↓reduceIte, reservedTotal]
        rw [Budget.add_assoc, total]

theorem erasing_cannot_add_reservations (leases : List Lease) (id : LeaseId) :
    Budget.Fits (reservedTotal (eraseLease leases id)) (reservedTotal leases) := by
  induction leases with
  | nil => simp [eraseLease, reservedTotal, Budget.Fits]
  | cons lease rest inductionHypothesis =>
      by_cases same : lease.id = id
      · simp [eraseLease, reservedTotal, same, Budget.add, Budget.Fits]
      · simp only [eraseLease, same, ↓reduceIte, reservedTotal, Budget.add]
        unfold Budget.Fits at inductionHypothesis ⊢
        exact ⟨Nat.add_le_add_left inductionHypothesis.1 _,
          Nat.add_le_add_left inductionHypothesis.2 _⟩

def Runtime.leaseStatus (runtime : Runtime) (id : LeaseId) : Option LeaseStatus :=
  if (findLease runtime.active id).isSome then
    some .active
  else if id ∈ runtime.committed then
    some .committed
  else if id ∈ runtime.aborted then
    some .aborted
  else
    none

theorem leaseStatus_none_implies_inactive
    (runtime : Runtime)
    (id : LeaseId)
    (fresh : runtime.leaseStatus id = none) :
    findLease runtime.active id = none := by
  cases found : findLease runtime.active id with
  | none => rfl
  | some lease => simp [Runtime.leaseStatus, found] at fresh

theorem leaseStatus_none_implies_terminal_fresh
    (runtime : Runtime)
    (id : LeaseId)
    (fresh : runtime.leaseStatus id = none) :
    id ∉ runtime.committed ∧ id ∉ runtime.aborted := by
  cases found : findLease runtime.active id with
  | some lease => simp [Runtime.leaseStatus, found] at fresh
  | none =>
      by_cases wasCommitted : id ∈ runtime.committed
      · simp [Runtime.leaseStatus, found, wasCommitted] at fresh
      · by_cases wasAborted : id ∈ runtime.aborted
        · simp [Runtime.leaseStatus, found, wasCommitted, wasAborted] at fresh
        · exact ⟨wasCommitted, wasAborted⟩

/-- Admit a fresh lease only when its complete reservation fits. -/
def prepare
    (runtime : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload) : Option Runtime :=
  if runtime.leaseStatus id = none then
    if Budget.Fits need runtime.available then
      let lease : Lease :=
        { id := id
          owner := runtime.id
          session := session
          baseVersion := runtime.version session
          reserved := need
          payload := payload }
      some { runtime with active := lease :: runtime.active }
    else
      none
  else
    none

/-- Commit resolves a lease from the issuer's active ledger and checks its revision. -/
def commit (runtime : Runtime) (id : LeaseId) : Option Runtime :=
  match findLease runtime.active id with
  | none => none
  | some lease =>
      if lease.owner = runtime.id then
        if runtime.version lease.session = lease.baseVersion then
          some
            { runtime with
              version := setVersion runtime.version lease.session (lease.baseVersion + 1)
              active := eraseLease runtime.active id
              committed := id :: runtime.committed }
        else
          none
      else
        none

/-- Abort resolves and removes one active lease. A terminal lease cannot run again. -/
def abort (runtime : Runtime) (id : LeaseId) : Option Runtime :=
  match findLease runtime.active id with
  | none => none
  | some lease =>
      if lease.owner = runtime.id then
        some
          { runtime with
            active := eraseLease runtime.active id
            aborted := id :: runtime.aborted }
      else
        none

theorem stale_commit_rejected
    (runtime : Runtime)
    (id : LeaseId)
    (lease : Lease)
    (active : findLease runtime.active id = some lease)
    (stale : runtime.version lease.session ≠ lease.baseVersion) :
    commit runtime id = none := by
  simp [commit, active, stale]

theorem accepted_prepare_fits
    (runtime prepared : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (accepted : prepare runtime id session need payload = some prepared) :
    Budget.Fits need runtime.available := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · exact fits
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

/-- Preparation records a runtime-issued lease as active under its fresh ID. -/
theorem accepted_prepare_records_owned_active_lease
    (runtime prepared : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (accepted : prepare runtime id session need payload = some prepared) :
    ∃ lease,
      findLease prepared.active id = some lease ∧
      lease.owner = prepared.id ∧
      prepared.leaseStatus id = some .active := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · simp [prepare, fresh, fits] at accepted
      subst prepared
      refine ⟨
        { id := id
          owner := runtime.id
          session := session
          baseVersion := runtime.version session
          reserved := need
          payload := payload }, ?_, ?_, ?_⟩
      · simp [findLease]
      · rfl
      · simp [Runtime.leaseStatus, findLease]
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

/-- Preparation determines every field of the admitted lease and no other state. -/
theorem accepted_prepare_exact
    (runtime prepared : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (accepted : prepare runtime id session need payload = some prepared) :
    prepared =
      { runtime with
        active :=
          { id := id
            owner := runtime.id
            session := session
            baseVersion := runtime.version session
            reserved := need
            payload := payload } :: runtime.active } := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · simp [prepare, fresh, fits] at accepted
      exact accepted.symm
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

theorem reused_lease_id_rejected
    (runtime : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (used : runtime.leaseStatus id ≠ none) :
    prepare runtime id session need payload = none := by
  simp [prepare, used]

theorem prepare_preserves_ledger_fit
    (runtime prepared : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (before : runtime.LedgerFits)
    (accepted : prepare runtime id session need payload = some prepared) :
    prepared.LedgerFits := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · simp [prepare, fresh, fits] at accepted
      subst prepared
      unfold Runtime.LedgerFits at before ⊢
      unfold Runtime.available Budget.Fits at fits
      unfold Budget.Fits at before ⊢
      simp only [Budget.reserve] at fits
      simp only [reservedTotal, Budget.add]
      constructor <;> omega
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

theorem commit_preserves_ledger_fit
    (runtime committedRuntime : Runtime)
    (id : LeaseId)
    (before : runtime.LedgerFits)
    (accepted : commit runtime id = some committedRuntime) :
    committedRuntime.LedgerFits := by
  unfold commit at accepted
  split at accepted <;> rename_i found
  · contradiction
  · split at accepted
    · split at accepted
      · simp at accepted
        subst committedRuntime
        exact Budget.fits_trans
          (erasing_cannot_add_reservations runtime.active id)
          before
      · contradiction
    · contradiction

theorem abort_preserves_ledger_fit
    (runtime abortedRuntime : Runtime)
    (id : LeaseId)
    (before : runtime.LedgerFits)
    (accepted : abort runtime id = some abortedRuntime) :
    abortedRuntime.LedgerFits := by
  unfold abort at accepted
  split at accepted <;> rename_i found
  · contradiction
  · split at accepted
    · simp at accepted
      subst abortedRuntime
      exact Budget.fits_trans
        (erasing_cannot_add_reservations runtime.active id)
        before
    · contradiction

/-- The only trusted root state. No lease or terminal ID exists initially. -/
def initialRuntime
    (id : RuntimeId)
    (capacity : Budget)
    (version : SessionId → Revision) : Runtime :=
  { id := id
    capacity := capacity
    version := version
    active := []
    committed := []
    aborted := [] }

/-- Valid states are exactly those produced from a canonical empty state. -/
inductive Valid : Runtime → Prop where
  | initial
      (id : RuntimeId)
      (capacity : Budget)
      (version : SessionId → Revision) :
      Valid (initialRuntime id capacity version)
  | prepare
      (before after : Runtime)
      (id : LeaseId)
      (session : SessionId)
      (need : Budget)
      (payload : Payload)
      (valid : Valid before)
      (step : prepare before id session need payload = some after) :
      Valid after
  | commit
      (before after : Runtime)
      (id : LeaseId)
      (valid : Valid before)
      (step : commit before id = some after) :
      Valid after
  | abort
      (before after : Runtime)
      (id : LeaseId)
      (valid : Valid before)
      (step : abort before id = some after) :
      Valid after

theorem valid_reservations_within_capacity
    (runtime : Runtime)
    (valid : Valid runtime) :
    runtime.LedgerFits := by
  induction valid with
  | initial => simp [initialRuntime, Runtime.LedgerFits, reservedTotal, Budget.Fits]
  | prepare before after id session need payload _ step inductionHypothesis =>
      exact prepare_preserves_ledger_fit
        before after id session need payload inductionHypothesis step
  | commit before after id _ step inductionHypothesis =>
      exact commit_preserves_ledger_fit before after id inductionHypothesis step
  | abort before after id _ step inductionHypothesis =>
      exact abort_preserves_ledger_fit before after id inductionHypothesis step

/-- Valid states have no duplicate active lease IDs. -/
theorem valid_active_ids_unique
    (runtime : Runtime)
    (valid : Valid runtime) :
    (activeIds runtime.active).Nodup := by
  induction valid with
  | initial => simp [initialRuntime, activeIds]
  | prepare before after id session need payload _ step inductionHypothesis =>
      by_cases fresh : before.leaseStatus id = none
      · by_cases fits : Budget.Fits need before.available
        · simp [prepare, fresh, fits] at step
          subst after
          have absent : id ∉ activeIds before.active := by
            rw [← findLease_none_iff_id_not_active]
            exact leaseStatus_none_implies_inactive before id fresh
          simpa [activeIds, absent] using List.nodup_cons.mpr ⟨absent, inductionHypothesis⟩
        · simp [prepare, fresh, fits] at step
      · simp [prepare, fresh] at step
  | commit before after id _ step inductionHypothesis =>
      unfold commit at step
      split at step <;> rename_i found
      · contradiction
      · split at step
        · split at step
          · simp at step
            subst after
            rw [activeIds_eraseLease]
            exact inductionHypothesis.erase id
          · contradiction
        · contradiction
  | abort before after id _ step inductionHypothesis =>
      unfold abort at step
      split at step <;> rename_i found
      · contradiction
      · split at step
        · simp at step
          subst after
          rw [activeIds_eraseLease]
          exact inductionHypothesis.erase id
        · contradiction

/-- Every active lease in a valid state was issued by that runtime. -/
theorem valid_active_leases_owned
    (runtime : Runtime)
    (valid : Valid runtime) :
    ∀ lease ∈ runtime.active, lease.owner = runtime.id := by
  induction valid with
  | initial => simp [initialRuntime]
  | prepare before after id session need payload _ step inductionHypothesis =>
      by_cases fresh : before.leaseStatus id = none
      · by_cases fits : Budget.Fits need before.available
        · simp [prepare, fresh, fits] at step
          subst after
          intro lease member
          simp at member
          rcases member with rfl | member
          · rfl
          · exact inductionHypothesis lease member
        · simp [prepare, fresh, fits] at step
      · simp [prepare, fresh] at step
  | commit before after id _ step inductionHypothesis =>
      unfold commit at step
      split at step <;> rename_i found
      · contradiction
      · split at step
        · split at step
          · simp at step
            subst after
            intro lease member
            exact inductionHypothesis lease (eraseLease_subset before.active id member)
          · contradiction
        · contradiction
  | abort before after id _ step inductionHypothesis =>
      unfold abort at step
      split at step <;> rename_i found
      · contradiction
      · split at step
        · simp at step
          subst after
          intro lease member
          exact inductionHypothesis lease (eraseLease_subset before.active id member)
        · contradiction

/-- Terminal IDs can never be active again in any valid future state. -/
theorem valid_terminal_ids_inactive
    (runtime : Runtime)
    (valid : Valid runtime) :
    ∀ id,
      id ∈ runtime.committed ∨ id ∈ runtime.aborted →
      findLease runtime.active id = none := by
  induction valid with
  | initial => simp [initialRuntime, findLease]
  | prepare before after id session need payload validBefore step inductionHypothesis =>
      by_cases fresh : before.leaseStatus id = none
      · by_cases fits : Budget.Fits need before.available
        · simp [prepare, fresh, fits] at step
          subst after
          have terminalFresh := leaseStatus_none_implies_terminal_fresh before id fresh
          intro sought terminal
          by_cases same : id = sought
          · subst sought
            rcases terminal with committed | aborted
            · exact False.elim (terminalFresh.1 committed)
            · exact False.elim (terminalFresh.2 aborted)
          · have oldAbsent := inductionHypothesis sought terminal
            simp [findLease, same, oldAbsent]
        · simp [prepare, fresh, fits] at step
      · simp [prepare, fresh] at step
  | commit before after id validBefore step inductionHypothesis =>
      have unique := valid_active_ids_unique before validBefore
      unfold commit at step
      split at step <;> rename_i found
      · contradiction
      · rename_i lease
        split at step
        · split at step
          · simp at step
            subst after
            intro sought terminal
            by_cases same : sought = id
            · subst sought
              exact eraseLease_removes_id_when_unique before.active id unique
            · have oldTerminal :
                  sought ∈ before.committed ∨ sought ∈ before.aborted := by
                  simpa [same] using terminal
              exact eraseLease_preserves_absence before.active id sought
                (inductionHypothesis sought oldTerminal)
          · contradiction
        · contradiction
  | abort before after id validBefore step inductionHypothesis =>
      have unique := valid_active_ids_unique before validBefore
      unfold abort at step
      split at step <;> rename_i found
      · contradiction
      · rename_i lease
        split at step
        · simp at step
          subst after
          intro sought terminal
          by_cases same : sought = id
          · subst sought
            exact eraseLease_removes_id_when_unique before.active id unique
          · have oldTerminal :
                sought ∈ before.committed ∨ sought ∈ before.aborted := by
                simpa [same] using terminal
            exact eraseLease_preserves_absence before.active id sought
              (inductionHypothesis sought oldTerminal)
        · contradiction

/-- Any terminal ID remains single-use across every valid continuation. -/
theorem valid_terminal_id_rejects_all_transitions
    (runtime : Runtime)
    (valid : Valid runtime)
    (id : LeaseId)
    (terminal : id ∈ runtime.committed ∨ id ∈ runtime.aborted) :
    commit runtime id = none ∧
      abort runtime id = none ∧
      ∀ session need payload,
        prepare runtime id session need payload = none := by
  have absent := valid_terminal_ids_inactive runtime valid id terminal
  constructor
  · simp [commit, absent]
  · constructor
    · simp [abort, absent]
    · intro session need payload
      rcases terminal with committed | aborted
      · simp [prepare, Runtime.leaseStatus, absent, committed]
      · by_cases committed : id ∈ runtime.committed <;>
          simp [prepare, Runtime.leaseStatus, absent, committed, aborted]

/-- Exact conservation: available plus active reservations equals total capacity. -/
theorem valid_exact_capacity_conservation
    (runtime : Runtime)
    (valid : Valid runtime) :
    Budget.add runtime.available (reservedTotal runtime.active) = runtime.capacity := by
  have fits := valid_reservations_within_capacity runtime valid
  rcases runtime with ⟨runtimeId, ⟨capacityMemory, capacityCompute⟩,
    versions, activeLeases, committedIds, abortedIds⟩
  unfold Runtime.LedgerFits Budget.Fits at fits
  simp [Runtime.available, Budget.reserve, Budget.add] at fits ⊢
  constructor <;> omega

/-- Aborting a newly admitted lease restores the exact previous availability. -/
theorem abort_after_prepare_restores_reservation
    (runtime prepared : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (accepted : prepare runtime id session need payload = some prepared) :
    ∃ abortedRuntime,
      abort prepared id = some abortedRuntime ∧
      abortedRuntime.available = runtime.available := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · simp [prepare, fresh, fits] at accepted
      subst prepared
      refine ⟨{ runtime with aborted := id :: runtime.aborted }, ?_, ?_⟩
      · simp [abort, findLease, eraseLease]
      · rfl
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

/-- The same newly admitted lease cannot be aborted twice. -/
theorem second_abort_rejected
    (runtime prepared firstAbort : Runtime)
    (id : LeaseId)
    (session : SessionId)
    (need : Budget)
    (payload : Payload)
    (accepted : prepare runtime id session need payload = some prepared)
    (first : abort prepared id = some firstAbort) :
    abort firstAbort id = none := by
  by_cases fresh : runtime.leaseStatus id = none
  · by_cases fits : Budget.Fits need runtime.available
    · simp [prepare, fresh, fits] at accepted
      subst prepared
      simp [abort, findLease, eraseLease] at first
      subst firstAbort
      have absent : findLease runtime.active id = none :=
        leaseStatus_none_implies_inactive runtime id fresh
      simp [abort, absent]
    · simp [prepare, fresh, fits] at accepted
  · simp [prepare, fresh] at accepted

/-- A successful commit consumed an active, issuer-owned, current-version lease. -/
theorem successful_commit_was_valid
    (runtime committedRuntime : Runtime)
    (id : LeaseId)
    (accepted : commit runtime id = some committedRuntime) :
    ∃ lease,
      findLease runtime.active id = some lease ∧
      lease.owner = runtime.id ∧
      runtime.version lease.session = lease.baseVersion := by
  unfold commit at accepted
  split at accepted <;> rename_i found
  · contradiction
  · split at accepted <;> rename_i owner
    · split at accepted <;> rename_i current
      · exact ⟨_, found, owner, current⟩
      · contradiction
    · contradiction

/-- Commit is terminal: the same valid lease cannot commit, abort, or prepare again. -/
theorem successful_commit_is_single_use
    (runtime committedRuntime : Runtime)
    (id : LeaseId)
    (valid : Valid runtime)
    (accepted : commit runtime id = some committedRuntime) :
    commit committedRuntime id = none ∧
      abort committedRuntime id = none ∧
      ∀ session need payload,
        prepare committedRuntime id session need payload = none := by
  have unique := valid_active_ids_unique runtime valid
  unfold commit at accepted
  split at accepted <;> rename_i found
  · contradiction
  · rename_i lease
    split at accepted <;> rename_i owner
    · split at accepted <;> rename_i current
      · simp at accepted
        subst committedRuntime
        have absent : findLease (eraseLease runtime.active id) id = none :=
          eraseLease_removes_id_when_unique runtime.active id unique
        constructor
        · simp [commit, absent]
        · constructor
          · simp [abort, absent]
          · intro session need payload
            simp [prepare, Runtime.leaseStatus, absent]
      · contradiction
    · contradiction

/-- Abort is terminal: the same valid lease cannot abort, commit, or prepare again. -/
theorem successful_abort_is_single_use
    (runtime abortedRuntime : Runtime)
    (id : LeaseId)
    (valid : Valid runtime)
    (accepted : abort runtime id = some abortedRuntime) :
    abort abortedRuntime id = none ∧
      commit abortedRuntime id = none ∧
      ∀ session need payload,
        prepare abortedRuntime id session need payload = none := by
  have unique := valid_active_ids_unique runtime valid
  unfold abort at accepted
  split at accepted <;> rename_i found
  · contradiction
  · rename_i lease
    split at accepted <;> rename_i owner
    · simp at accepted
      subst abortedRuntime
      have absent : findLease (eraseLease runtime.active id) id = none :=
        eraseLease_removes_id_when_unique runtime.active id unique
      constructor
      · simp [abort, absent]
      · constructor
        · simp [commit, absent]
        · intro session need payload
          by_cases wasCommitted : id ∈ runtime.committed <;>
            simp [prepare, Runtime.leaseStatus, absent, wasCommitted]
    · contradiction

/-- Algebraic helper for two active, current leases on distinct sessions. -/
theorem active_pair_commits_commute
    (runtime : Runtime)
    (left right : Lease)
    (leftOwner : left.owner = runtime.id)
    (rightOwner : right.owner = runtime.id)
    (differentIds : left.id ≠ right.id)
    (differentSessions : left.session ≠ right.session)
    (leftCurrent : runtime.version left.session = left.baseVersion)
    (rightCurrent : runtime.version right.session = right.baseVersion) :
    let admitted : Runtime := { runtime with active := left :: right :: runtime.active }
    ∃ afterLeft afterLeftRight afterRight afterRightLeft,
      commit admitted left.id = some afterLeft ∧
      commit afterLeft right.id = some afterLeftRight ∧
      commit admitted right.id = some afterRight ∧
      commit afterRight left.id = some afterRightLeft ∧
      afterLeftRight.version = afterRightLeft.version ∧
      afterLeftRight.available = afterRightLeft.available ∧
      afterLeftRight.active = afterRightLeft.active ∧
      (∀ id, id ∈ afterLeftRight.committed ↔ id ∈ afterRightLeft.committed) ∧
      afterLeftRight.aborted = afterRightLeft.aborted := by
  dsimp
  have reverseIds : right.id ≠ left.id := Ne.symm differentIds
  have reverseSessions : right.session ≠ left.session := Ne.symm differentSessions
  have rightStillCurrent :
      setVersion runtime.version left.session (left.baseVersion + 1) right.session =
        right.baseVersion := by
    simp [setVersion, reverseSessions, rightCurrent]
  have leftStillCurrent :
      setVersion runtime.version right.session (right.baseVersion + 1) left.session =
        left.baseVersion := by
    simp [setVersion, differentSessions, leftCurrent]
  let afterLeft : Runtime :=
    { runtime with
      version := setVersion runtime.version left.session (left.baseVersion + 1)
      active := right :: runtime.active
      committed := left.id :: runtime.committed }
  let afterLeftRight : Runtime :=
    { runtime with
      version :=
        setVersion
          (setVersion runtime.version left.session (left.baseVersion + 1))
          right.session
          (right.baseVersion + 1)
      active := runtime.active
      committed := right.id :: left.id :: runtime.committed }
  let afterRight : Runtime :=
    { runtime with
      version := setVersion runtime.version right.session (right.baseVersion + 1)
      active := left :: runtime.active
      committed := right.id :: runtime.committed }
  let afterRightLeft : Runtime :=
    { runtime with
      version :=
        setVersion
          (setVersion runtime.version right.session (right.baseVersion + 1))
          left.session
          (left.baseVersion + 1)
      active := runtime.active
      committed := left.id :: right.id :: runtime.committed }
  refine ⟨afterLeft, afterLeftRight, afterRight, afterRightLeft,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [commit, findLease, eraseLease, leftOwner, leftCurrent, afterLeft]
  · simp [commit, findLease, eraseLease, rightOwner, rightStillCurrent,
      afterLeft, afterLeftRight]
  · simp [commit, findLease, eraseLease, rightOwner, rightCurrent, differentIds, afterRight]
  · simp [commit, findLease, eraseLease, leftOwner, leftStillCurrent,
      afterRight, afterRightLeft]
  · funext session
    by_cases isLeft : session = left.session
    · subst session
      simp [afterLeftRight, afterRightLeft, setVersion, differentSessions]
    · by_cases isRight : session = right.session
      · subst session
        simp [afterLeftRight, afterRightLeft, setVersion, reverseSessions]
      · simp [afterLeftRight, afterRightLeft, setVersion, isLeft, isRight]
  · rfl
  · rfl
  · intro id
    simp [afterLeftRight, afterRightLeft]
    constructor
    · intro member
      rcases member with isRight | isLeft | old
      · exact Or.inr (Or.inl isRight)
      · exact Or.inl isLeft
      · exact Or.inr (Or.inr old)
    · intro member
      rcases member with isLeft | isRight | old
      · exact Or.inr (Or.inl isLeft)
      · exact Or.inl isRight
      · exact Or.inr (Or.inr old)
  · rfl

/-- Two leases admitted by actual prepare steps commute when their sessions differ. -/
theorem independently_prepared_commits_commute
    (runtime afterFirstPrepare admitted : Runtime)
    (leftId rightId : LeaseId)
    (leftSession rightSession : SessionId)
    (leftNeed rightNeed : Budget)
    (leftPayload rightPayload : Payload)
    (valid : Valid runtime)
    (differentIds : leftId ≠ rightId)
    (differentSessions : leftSession ≠ rightSession)
    (firstPrepare :
      prepare runtime leftId leftSession leftNeed leftPayload =
        some afterFirstPrepare)
    (secondPrepare :
      prepare afterFirstPrepare rightId rightSession rightNeed rightPayload =
        some admitted) :
    ∃ afterRight afterRightLeft afterLeft afterLeftRight,
      commit admitted rightId = some afterRight ∧
      commit afterRight leftId = some afterRightLeft ∧
      commit admitted leftId = some afterLeft ∧
      commit afterLeft rightId = some afterLeftRight ∧
      Valid afterRightLeft ∧
      Valid afterLeftRight ∧
      afterRightLeft.version = afterLeftRight.version ∧
      afterRightLeft.available = afterLeftRight.available ∧
      afterRightLeft.active = afterLeftRight.active ∧
      (∀ id, id ∈ afterRightLeft.committed ↔ id ∈ afterLeftRight.committed) ∧
      afterRightLeft.aborted = afterLeftRight.aborted := by
  have firstExact := accepted_prepare_exact
    runtime afterFirstPrepare leftId leftSession leftNeed leftPayload firstPrepare
  subst afterFirstPrepare
  have secondExact := accepted_prepare_exact
    { runtime with
      active :=
        { id := leftId
          owner := runtime.id
          session := leftSession
          baseVersion := runtime.version leftSession
          reserved := leftNeed
          payload := leftPayload } :: runtime.active }
    admitted rightId rightSession rightNeed rightPayload secondPrepare
  subst admitted
  let leftLease : Lease :=
    { id := leftId
      owner := runtime.id
      session := leftSession
      baseVersion := runtime.version leftSession
      reserved := leftNeed
      payload := leftPayload }
  let rightLease : Lease :=
    { id := rightId
      owner := runtime.id
      session := rightSession
      baseVersion := runtime.version rightSession
      reserved := rightNeed
      payload := rightPayload }
  have commuting := active_pair_commits_commute
    runtime rightLease leftLease
    (by rfl) (by rfl)
    (Ne.symm differentIds) (Ne.symm differentSessions)
    (by rfl) (by rfl)
  simp only [rightLease, leftLease] at commuting
  rcases commuting with
    ⟨afterRight, afterRightLeft, afterLeft, afterLeftRight,
      commitRight, commitRightLeft, commitLeft, commitLeftRight,
      sameVersion, sameAvailable, sameActive, sameCommitted, sameAborted⟩
  have validAfterFirstPrepare : Valid
      { runtime with active := leftLease :: runtime.active } := by
    exact Valid.prepare runtime _ leftId leftSession leftNeed leftPayload valid firstPrepare
  have validAdmitted : Valid
      { runtime with active := rightLease :: leftLease :: runtime.active } := by
    exact Valid.prepare _ _ rightId rightSession rightNeed rightPayload
      validAfterFirstPrepare secondPrepare
  have validAfterRight : Valid afterRight :=
    Valid.commit _ afterRight rightId validAdmitted commitRight
  have validAfterRightLeft : Valid afterRightLeft :=
    Valid.commit afterRight afterRightLeft leftId validAfterRight commitRightLeft
  have validAfterLeft : Valid afterLeft :=
    Valid.commit _ afterLeft leftId validAdmitted commitLeft
  have validAfterLeftRight : Valid afterLeftRight :=
    Valid.commit afterLeft afterLeftRight rightId validAfterLeft commitLeftRight
  exact ⟨afterRight, afterRightLeft, afterLeft, afterLeftRight,
    commitRight, commitRightLeft, commitLeft, commitLeftRight,
    validAfterRightLeft, validAfterLeftRight,
    sameVersion, sameAvailable, sameActive, sameCommitted, sameAborted⟩

end WhisperRuntimeFormal
