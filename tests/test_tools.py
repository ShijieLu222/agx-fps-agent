import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.pack import pack
from scripts.replay_check import check, select_seat, validate_action


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def log(self, rows):
        path = self.root / "training.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def test_pack_is_reproducible_and_only_includes_agent(self):
        source = self.root / "agent.py"
        source.write_text("def act(o): return {}\n")
        (self.root / ".env").write_text("DO_NOT_PACK=secret")
        first, second = self.root / "first.zip", self.root / "second.zip"
        self.assertEqual(pack(source, first), pack(source, second))
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.namelist(), ["agent.py"])
            self.assertEqual(archive.read("agent.py"), source.read_bytes())

    def test_pack_refuses_invalid_python(self):
        source = self.root / "agent.py"
        source.write_text("def broken(")
        with self.assertRaises(SyntaxError):
            pack(source, self.root / "bundle.zip")

    def test_incomplete_or_post_marker_data_rejected(self):
        decision = {"seat": 0, "observation": {}}
        for data in ([decision], [decision, {"type": "export_complete"}, {"type": "outcome"}]):
            with self.assertRaises(ValueError):
                select_seat(self.log(data))

    def test_multiple_seats_require_explicit_selection(self):
        path = self.log([{"seat": n, "observation": {}} for n in (0, 1)] + [{"type": "export_complete"}])
        with self.assertRaises(ValueError):
            select_seat(path)
        self.assertEqual(select_seat(path, 1), 1)

    def test_checks_only_selected_seat_and_calls_init(self):
        agent = self.root / "agent.py"
        agent.write_text("state = None\ndef init(context):\n global state\n state = context['seat']\ndef act(o):\n assert state == o['seat']\n return {'move_forward': 0.5}\n")
        path = self.log([
            {"seat": 0, "observation": {"seat": 0, "match_tick": 0, "map": {}}},
            {"seat": 1, "observation": {"seat": 1, "match_tick": 0, "map": {}}},
            {"seat": 0, "observation": {"seat": 0, "match_tick": 3}},
            {"type": "export_complete"},
        ])
        report = check(path, agent, 0)
        self.assertEqual(report["decisions_checked"], 2)
        self.assertEqual(report["errors"], [])

    def test_basic_contract_rejects_nonfinite_bool_numbers_and_extra_keys(self):
        for action in ({"yaw_rate": float("nan")}, {"weapon_slot": True},
                       {"move_forward": True}, {"fire": 1}, {"yaw_rate": 1.01},
                       {"unknown": 0}, {"use": "shoot"}):
            with self.assertRaises(ValueError):
                validate_action(action)
        self.assertEqual(validate_action({"buy": "ak47", "fire": False, "yaw_rate": -1}), [])


if __name__ == "__main__":
    unittest.main()
