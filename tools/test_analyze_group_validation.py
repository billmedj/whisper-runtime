"""Archive joins, fixed enrollment, and non-interference of group diagnostics."""

import contextlib
import copy
import hashlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import analyze_group_validation as replay

ROOT = Path(__file__).resolve().parents[1]
JFK = "evidence/modal-t4-tiny-en-word-alignment-v6-2026-09-05.json"


class GroupValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = {name: (ROOT / name).read_bytes() for name in replay.ARCHIVES}

    def test_fixed_full_histories_include_exclusions_and_repeated_states(self):
        report = replay.analyze_records(self.payloads)
        self.assertEqual(report, replay.analyze_records(self.payloads))
        for name, expected in (
            ("traces", 24),
            ("enrolled", 14),
            ("excluded", 10),
            ("distinct_anchor_states", 8),
        ):
            self.assertEqual(report["counts"][name], expected)
        self.assertEqual(len(report["cells"]), 3)
        for cell in report["cells"]:
            self.assertFalse(cell["publication_authorized"])
            for trace in cell["traces"]:
                if trace["enrollment"] == "enrolled":
                    self.assertFalse(trace["diagnostic"]["publication_authorized"])
                    self.assertIsNotNone(trace["anchor_origin"])
                else:
                    self.assertIsNone(trace["diagnostic"])
                    self.assertIn(
                        trace["exclusion_reason"],
                        {"no_committed_anchor", "insufficient_anchor"},
                    )

    def test_fixed_archive_bytes_are_checked_before_any_scoring(self):
        damaged = {**self.payloads, JFK: self.payloads[JFK] + b" "}
        with patch.object(replay, "diagnose_group_anchor") as scorer:
            with self.assertRaises(ValueError):
                replay.analyze_records(damaged)
            scorer.assert_not_called()

    def test_invalid_event_and_observation_joins_fail_before_scoring(self):
        for mutation in ("head", "commit", "native", "publication"):
            with self.subTest(mutation=mutation):
                record = json.loads(self.payloads[JFK])
                cell = record["cells"][1]
                if mutation == "head":
                    cell["decision_traces"][2]["committed_before_sample"] += 16
                elif mutation == "commit":
                    commit = next(
                        e for e in cell["events"] if e["kind"].lower() == "commit"
                    )
                    commit["committed_through_sample"] += 16
                elif mutation == "native":
                    cell["decision_traces"][2]["result"]["window_id"] = (
                        "another-session"
                    )
                else:
                    publication = next(
                        t["word_publication"]
                        for t in cell["decision_traces"]
                        if t["word_publication"] is not None
                    )
                    publication["text"] = "invented committed text"
                changed = json.dumps(record).encode()
                payloads = {**self.payloads, JFK: changed}
                # Bypass only the digest for this synthetic corruption. Exercise
                # the state/COMMIT joins, not acceptance of a different archive.
                with (
                    patch.dict(
                        replay.ARCHIVES, {JFK: hashlib.sha256(changed).hexdigest()}
                    ),
                    patch.object(replay, "diagnose_group_anchor") as scorer,
                ):
                    with self.assertRaises((ValueError, TypeError)):
                        replay.analyze_records(payloads)
                    scorer.assert_not_called()

    def test_shadow_results_cannot_change_later_frozen_state(self):
        original = replay.diagnose_group_anchor

        def changed(status):
            def score(*args, **kwargs):
                result = original(*args, **kwargs)
                return {**result, "status": status}

            return score

        reports = []
        for status in ("matched", "relocated"):
            with patch.object(
                replay, "diagnose_group_anchor", side_effect=changed(status)
            ):
                reports.append(replay.analyze_records(self.payloads))
        for left, right in zip(reports[0]["cells"], reports[1]["cells"]):
            for before, after in zip(left["traces"], right["traces"]):
                frozen_before, frozen_after = (
                    copy.deepcopy(before),
                    copy.deepcopy(after),
                )
                frozen_before.pop("diagnostic")
                frozen_after.pop("diagnostic")
                self.assertEqual(frozen_before, frozen_after)

    def test_archived_results_replay_without_changing_recorded_provenance(self):
        payload = (ROOT / "evidence/group-validation-2026-09-06.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "356a9fcf14771676ce829043d6e637bf7fc99c841d0d1bd69662b869111973b5",
        )
        archived = json.loads(payload)
        actual = json.loads(json.dumps(replay.analyze_records(self.payloads)))
        # Analysis sources can evolve; preserve their original hashes in the
        # archive and compare all saved outcomes and input identities.
        archived.pop("source_sha256")
        actual.pop("source_sha256")
        self.assertEqual(actual, archived)
        self.assertEqual(actual["counts"]["group_statuses"], {"matched": 14})
        self.assertEqual(actual["counts"]["strict_statuses"], {"matched": 14})

    def test_cli_creates_once_and_does_not_overwrite_evidence(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            self.assertEqual(replay.main(["--output", str(output)]), 0)
            saved = output.read_bytes()
            self.assertEqual(
                json.loads(saved),
                json.loads(json.dumps(replay.analyze_records(self.payloads))),
            )
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    replay.main(["--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
