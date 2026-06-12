from __future__ import annotations

from pathlib import Path

from isaac_fight.tasks.direct.unitree_1v1.opponent_pool import OpponentPool


def test_pool_add_save_load_and_sample(tmp_path: Path):
    ckpt = tmp_path / "policy_v000001.pt"
    ckpt.write_bytes(b"PK\x03\x04dummy")
    pool = OpponentPool(tmp_path / "pool")
    record = pool.add_checkpoint(ckpt, version=1, elo=1010, tags=("torchscript",))
    assert record.policy_id == "policy_v000001"
    pool.update_result(record.policy_id, 0.0, elo=990)

    reloaded = OpponentPool(tmp_path / "pool")
    assert len(reloaded) == 1
    sample = reloaded.sample(active_elo=1000)
    assert sample is not None
    assert sample.policy.policy_id == record.policy_id
    assert 0.0 < sample.probability <= 1.0


def test_pool_ids_do_not_collide_for_same_checkpoint_number(tmp_path: Path):
    ckpt_a = tmp_path / "run_a" / "agent_000050.pt"
    ckpt_b = tmp_path / "run_b" / "agent_000050.pt"
    ckpt_a.parent.mkdir()
    ckpt_b.parent.mkdir()
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")

    pool = OpponentPool(tmp_path / "pool")
    first = pool.add_checkpoint(ckpt_a, version=50)
    second = pool.add_checkpoint(ckpt_b, version=50)

    assert first.policy_id == "policy_v000050"
    assert second.policy_id != first.policy_id
    assert len(pool) == 2
