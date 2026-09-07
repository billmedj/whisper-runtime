"""CPU-only falsification replay; no planner, cache, inference or policy change.

Hashes identify diagnostic inputs, not independent acoustic proof or execution
attestation. Archive mutations show preserved refusals, not individual gate
causality. Only the clean synthetic controls isolate those gates. Duplicate
evaluation shows pure-function idempotence, not production deduplication.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import unittest
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import verify_continuity_witness as replay
from whisper_runtime import AudioSpan

ROOT = replay.p.ROOT
FIXTURES = {
    "same_origin_reuse_and_fresh_growth": "test_same_origin_commit_keeps_exact_accepted_witness_without_rewind",
    "checkpoint_reuse_without_redecode": "test_checkpoint_derives_same_witness_without_new_cache_fields_or_redecode",
    "rebase_requires_fresh_pair": "test_rebase_still_starts_a_fresh_growing_pair",
}


@contextmanager
def fixture_module(name):
    """Scope named imports and fixtures' string-based mock targets together."""
    names = (
        "test_continuity_witness",
        "test_same_origin_word_commits",
        "test_continuous_endpoint_words",
        "test_audio_endpoints",
        "test_continuous_evidence",
        "test_continuous_stream",
    )
    previous_path = sys.path[:]
    previous_modules = {key: sys.modules[key] for key in names if key in sys.modules}
    try:
        sys.path.insert(0, str(ROOT / "tests"))
        for key in names:
            sys.modules.pop(key, None)
        yield importlib.import_module(name)
    finally:
        sys.path[:] = previous_path
        for key in names:
            sys.modules.pop(key, None)
        sys.modules.update(previous_modules)


def fingerprint(arguments):
    return replay.p.shared._hash(
        {
            k: replay.p.shared._sha(v) if isinstance(v, bytes) else asdict(v)
            for k, v in arguments.items()
        }
    )


def fixture_checks():
    outcomes = {}
    with fixture_module("test_same_origin_word_commits") as module:
        for label, method in FIXTURES.items():
            result = unittest.TestResult()
            module.SameOriginWordCommitTests(method).run(result)
            replay.p._require(
                result.wasSuccessful() and not result.skipped,
                "existing fixture failed: " + label,
            )
            outcomes[label] = "passed"
    return outcomes


def counterfactual_inputs(arguments):
    """Mutate immutable observations in memory; never replace archived receipts."""
    candidate, identity = arguments["candidate"], arguments["candidate_identity"]
    pcm = bytearray(arguments["pcm"])
    pcm[candidate.native.start_ms * 32] ^= 1
    end = candidate.native.end_ms + 20
    overlap_index = max(
        i
        for i, w in enumerate(candidate.words)
        if w.span.start_ms < arguments["joint"].native.end_ms
    )
    words = list(candidate.words)
    word = words[overlap_index]
    words[overlap_index] = replace(
        word, span=AudioSpan(word.span.start_ms, word.span.end_ms - 20)
    )
    return {
        "changed_pcm": {"pcm": bytes(pcm)},
        "changed_model_identity": {
            "candidate_identity": replace(
                identity,
                declared_model=replace(identity.declared_model, fingerprint="0" * 64),
            )
        },
        "changed_decode_options": {
            "candidate_identity": replace(
                identity,
                requested_decode_options=replace(
                    identity.requested_decode_options, suppress_blank=False
                ),
            )
        },
        "changed_window_origin": {
            "candidate": replace(
                candidate,
                native=replace(
                    candidate.native,
                    start_ms=candidate.native.start_ms - 20,
                    analysis_span=AudioSpan(
                        candidate.native.start_ms - 20, candidate.native.end_ms
                    ),
                ),
            )
        },
        "partial_word_still_crosses_overlap": {
            "candidate": replace(candidate, words=tuple(words))
        },
        "extended_window_without_fresh_observation": {
            "pcm": arguments["pcm"] + b"\x01\x00" * 320,
            "candidate": replace(
                candidate,
                native=replace(
                    candidate.native,
                    end_ms=end,
                    analysis_span=AudioSpan(candidate.native.start_ms, end),
                ),
            ),
        },
    }


def clean_fixture_controls():
    with fixture_module("test_continuity_witness") as module:
        fixture = module.ContinuityWitnessTests()
    fixture.setUp()
    arguments = fixture.arguments
    before = fingerprint(arguments)
    baseline = fixture.assess()
    replay.p._require(baseline.reasons == (), "clean control already refused")
    expected = {
        "changed_pcm": ("joint_pcm_mismatch", "candidate_pcm_mismatch"),
        "changed_model_identity": (
            "analysis_identity_mismatch",
            "published_model_mismatch",
        ),
        "changed_decode_options": ("analysis_identity_mismatch",),
        "changed_window_origin": ("incompatible_observation_windows",),
    }
    cases = counterfactual_inputs(arguments)
    outcomes = {}
    for label, reasons in expected.items():
        changed = {**arguments, **cases[label]}
        result = replay.assess_continuity_witness(**changed)
        replay.p._require(
            result.reasons == reasons, "clean control gate differs: " + label
        )
        outcomes[label] = dict(
            input_sha256=fingerprint(changed), newly_present_reasons=result.reasons
        )
    replay.p._require(fingerprint(arguments) == before, "clean fixture mutated")
    return dict(
        baseline=dict(input_sha256=before, reasons=baseline.reasons),
        correspondence_only=True,
        mutations=outcomes,
    )


def verify(root=ROOT):
    arguments, provenance = replay.load_observations(root)
    before = fingerprint(arguments)
    original = replay.assess_continuity_witness(**arguments)
    duplicate = replay.assess_continuity_witness(**arguments)
    replay.p._require(
        original == duplicate and fingerprint(arguments) == before,
        "duplicate replay changed inputs or diagnostic",
    )
    outcomes = {}
    for label, changes in counterfactual_inputs(arguments).items():
        changed = {**arguments, **changes}
        result = replay.assess_continuity_witness(**changed)
        replay.p._require(bool(result.reasons), "counterexample unexpectedly passed")
        outcomes[label] = dict(
            input_sha256=fingerprint(changed), reasons=result.reasons
        )
    replay.p._require(fingerprint(arguments) == before, "archive input mutated")
    return dict(
        schema_version="1-offline-evidence-reuse-falsification",
        source_sha256={
            k: provenance[k]
            for k in (
                "paced_archive_sha256",
                "prompt_archive_sha256",
                "full_pcm_sha256",
            )
        },
        duplicate=dict(
            input_sha256=before,
            evaluations=2,
            identical_result=True,
            reasons=original.reasons,
            additional_acoustic_observations=0,
            interpretation="pure_function_diagnostic_idempotence_only",
            production_deduplication_demonstrated=False,
        ),
        archive_mutation_claim="preserved_refusals_not_individual_gate_causality",
        counterfactual_mutations=outcomes,
        clean_synthetic_controls=clean_fixture_controls(),
        existing_scripted_fixture_checks=fixture_checks(),
        shadow_planner_implemented=False,
        next_work_selection_validated=False,
        cache_implemented=False,
        native_inference_run=False,
        publication_authorized=False,
        audio_retirement_authorized=False,
        recognition_quality_claim=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args(argv).root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
