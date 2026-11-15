"""Corrupt/truncate real-shaped exports rather than trusting a completion label."""
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_training import audit, verify_export


RECORDED_BUNDLE = "a" * 64


def decision(step=0, seat=5):
    observation = {
        "seat": seat, "team": seat // 5, "role": 0, "rules_version": "rsi-fps-5.5",
        "match_tick": step * 3, "map_id": "test-map", "static_schema_hash": "test-geometry",
        "round_index": 0, "phase": "buy", "me": {"alive": True},
    }
    if step == 0:
        observation["map"] = {"sites": [], "nav": {}}
    return {"seat": seat, "step": step, "observation": observation,
            "action": {}, "applied_action": {}, "fault": None}


def export_bytes(decisions=None, participants=None):
    decisions = decisions if decisions is not None else [decision(), decision(1)]
    participants = participants if participants is not None else [
        {"seat": 5, "team": 1, "role": 0, "source_hash": RECORDED_BUNDLE}]
    metadata = {"type": "metadata", "schema": "agent-tournament.training-export.v2",
                "game_id": "fps", "rules_version": "rsi-fps-5.5", "complete": False,
                "participants": participants, "decision_encoding": "original-jsonl-bytes",
                "observation_semantics": "recorded_policy_input", "completion_record": "export_complete",
                "seed": 987654321}
    encoded = [(json.dumps(row) + "\n").encode() for row in decisions]
    raw = b"".join(encoded)
    footer = {"type": "export_complete", "complete": True, "rows": len(encoded),
              "raw_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
              "seat_calls": dict(Counter(str(row["seat"]) for row in decisions))}
    return ((json.dumps(metadata) + "\n").encode() + raw
            + (json.dumps({"type": "outcome", "winner": 1}) + "\n").encode()
            + (json.dumps(footer) + "\n").encode())


class ExportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "training.jsonl"
        self.policy = self.root / "agent.py"
        self.policy.write_text("def init(context):\n assert context['seed'] == 0\n"
                               "def act(o):\n return {}\n")
        self.gc_was_enabled = gc.isenabled()

    def tearDown(self):
        if self.gc_was_enabled:
            gc.enable()

    def write(self, data):
        self.path.write_bytes(data)
        return self.path

    def test_complete_export_preserves_source_identity_and_uses_public_seed(self):
        report = audit(self.write(export_bytes()), self.policy, 5, RECORDED_BUNDLE)
        self.assertEqual(report["decisions_checked"], 2)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["export_integrity"]["rows_bytes_sha256_seats_verified"])
        self.assertEqual(report["recorded_bundle_sha256"], RECORDED_BUNDLE)
        self.assertEqual(report["candidate_source_sha256"], hashlib.sha256(self.policy.read_bytes()).hexdigest())
        self.assertNotIn("987654321", json.dumps(report))

    def test_bundle_identity_mismatch_rejected_before_candidate_runs(self):
        self.policy.write_text("raise AssertionError('candidate must not execute')\n")
        with self.assertRaisesRegex(ValueError, "bundle"):
            audit(self.write(export_bytes()), self.policy, 5, "b" * 64)

    def test_payload_tamper_with_unchanged_completion_hash_rejected(self):
        # A structurally valid action modification still invalidates raw bytes.
        data = export_bytes().replace(b'"action": {}', b'"action": {"fire": true}', 1)
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_export(self.write(data))

    def test_truncated_missing_outcome_and_appended_data_rejected(self):
        lines = export_bytes().splitlines(keepends=True)
        cases = [b"".join(lines[:-1]), b"".join(lines[:-2] + lines[-1:]),
                 b"".join(lines) + lines[1], b"".join(lines) + lines[-1]]
        for data in cases:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                verify_export(self.write(data))

    def test_completion_counts_and_complete_flag_are_enforced(self):
        lines = export_bytes().splitlines(keepends=True)
        for field, value in (("rows", 99), ("raw_bytes", 1), ("sha256", "b" * 64),
                             ("seat_calls", {"5": 1}), ("complete", False)):
            footer = json.loads(lines[-1]); footer[field] = value
            data = b"".join(lines[:-1]) + (json.dumps(footer) + "\n").encode()
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_export(self.write(data))

    def test_valid_checksum_cannot_hide_observation_identity_changes(self):
        for field, value in (("seat", 4), ("team", 0), ("team", True), ("role", 1), ("role", False),
                             ("rules_version", "rsi-fps-other"), ("map_id", "other-map"),
                             ("match_tick", 0)):
            rows = [decision(), decision(1)]
            rows[1]["observation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_export(self.write(export_bytes(rows)))

    def test_nonfinite_observation_is_rejected_even_with_matching_checksum(self):
        rows = [decision(), decision(1)]
        rows[1]["observation"]["me"]["x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            verify_export(self.write(export_bytes(rows)))

    def test_metadata_game_and_duplicate_seat_rejected(self):
        data = export_bytes().replace(b'"game_id": "fps"', b'"game_id": "racing"')
        with self.assertRaises(ValueError):
            verify_export(self.write(data))
        participant = {"seat": 5, "team": 1, "role": 0, "source_hash": RECORDED_BUNDLE}
        with self.assertRaises(ValueError):
            verify_export(self.write(export_bytes(participants=[participant, copy.deepcopy(participant)])))

    def test_multi_seat_export_requires_selection(self):
        participants = [{"seat": seat, "team": seat // 5, "role": 0,
                         "source_hash": RECORDED_BUNDLE} for seat in (4, 5)]
        data = export_bytes([decision(0, 4), decision(0, 5)], participants)
        with self.assertRaisesRegex(ValueError, "multiple seats"):
            verify_export(self.write(data))
        self.assertEqual(verify_export(self.path, 5)[0], 5)

    def test_invalid_candidate_action_is_reported_and_replay_stops(self):
        self.policy.write_text("def act(o):\n return {'fire': 1}\n")
        report = audit(self.write(export_bytes()), self.policy)
        self.assertEqual(report["decisions_checked"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("fire", report["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
