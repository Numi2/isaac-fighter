from __future__ import annotations

from isaac_fight.tasks.direct.unitree_1v1.replay import MatchReplayRecorder, load_replay


def test_replay_jsonl_round_trip(tmp_path):
    path = tmp_path / "match.jsonl"
    with MatchReplayRecorder(path) as recorder:
        recorder.write_step(1, {"winner": 0, "fighter_a": {"reward": 1.0}})
        recorder.write_match_end({"winner": 1})
    records = load_replay(path)
    assert records[0]["type"] == "header"
    assert records[1]["type"] == "step"
    assert records[2]["type"] == "match_end"
