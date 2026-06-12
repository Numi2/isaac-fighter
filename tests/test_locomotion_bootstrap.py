from __future__ import annotations

from pathlib import Path

import torch
from isaac_fight.locomotion_bootstrap import (
    LOCOMOTION_WARMSTART_SCHEMA,
    apply_locomotion_warmstart,
    create_fight_warmstart,
    inspect_rsl_rl_checkpoint,
    is_locomotion_warmstart_checkpoint,
    sync_locomotion_artifact,
)
from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
from isaac_fight.tasks.direct.unitree_1v1.observations import observation_dim
from torch import nn


def _rsl_checkpoint(path: Path, obs_dim: int = 480, action_dim: int = 29) -> dict[str, torch.Tensor]:
    state = {
        "actor.0.weight": torch.full((512, obs_dim), 0.10),
        "actor.0.bias": torch.full((512,), 0.11),
        "actor.2.weight": torch.full((256, 512), 0.20),
        "actor.2.bias": torch.full((256,), 0.21),
        "actor.4.weight": torch.full((128, 256), 0.30),
        "actor.4.bias": torch.full((128,), 0.31),
        "actor.6.weight": torch.full((action_dim, 128), 0.40),
        "actor.6.bias": torch.full((action_dim,), 0.41),
        "critic.0.weight": torch.full((512, obs_dim), -0.10),
        "critic.0.bias": torch.full((512,), -0.11),
        "critic.2.weight": torch.full((256, 512), -0.20),
        "critic.2.bias": torch.full((256,), -0.21),
        "critic.4.weight": torch.full((128, 256), -0.30),
        "critic.4.bias": torch.full((128,), -0.31),
        "critic.6.weight": torch.full((1, 128), -0.40),
        "critic.6.bias": torch.full((1,), -0.41),
    }
    torch.save({"model_state_dict": state, "iter": 12, "infos": {"source": "unit-test"}}, path)
    return state


def test_inspect_rsl_rl_checkpoint_identifies_g1_layers(tmp_path: Path):
    checkpoint = tmp_path / "unitree_g1_velocity.pt"
    _rsl_checkpoint(checkpoint)

    info = inspect_rsl_rl_checkpoint(checkpoint)

    assert info.robot_name == "g1_29dof"
    assert info.source_task == "Unitree-G1-29dof-Velocity"
    assert info.obs_dim == 480
    assert info.action_dim == 29
    assert [layer.key for layer in info.actor_layers] == [
        "actor.0.weight",
        "actor.2.weight",
        "actor.4.weight",
        "actor.6.weight",
    ]


def test_warmstart_transfers_compatible_actor_layers(tmp_path: Path):
    source = tmp_path / "unitree_g1_velocity.pt"
    source_state = _rsl_checkpoint(source)
    output = tmp_path / "warmstarts" / "g1_velocity_to_fight.pt"

    report = create_fight_warmstart(source, output)

    assert output.exists()
    assert report.fight_agent == FIGHTER_A
    warmstart = torch.load(output, map_location="cpu", weights_only=False)
    assert warmstart["__metadata__"]["schema"] == LOCOMOTION_WARMSTART_SCHEMA
    assert warmstart["__metadata__"]["not_opponent"] is True
    assert is_locomotion_warmstart_checkpoint(output)

    g1_policy = warmstart[FIGHTER_A]["policy"]
    assert g1_policy["net_container.0.weight"].shape == (512, observation_dim(29))
    assert not torch.equal(g1_policy["net_container.0.weight"], source_state["actor.0.weight"])
    assert torch.equal(g1_policy["net_container.2.weight"], source_state["actor.2.weight"])
    assert torch.equal(g1_policy["net_container.4.weight"], source_state["actor.4.weight"])
    assert torch.equal(g1_policy["net_container.6.weight"], source_state["actor.6.weight"])
    assert torch.equal(g1_policy["net_container.6.bias"], source_state["actor.6.bias"])

    h1_policy = warmstart[FIGHTER_B]["policy"]
    assert h1_policy["net_container.6.weight"].shape == (19, 128)
    assert warmstart[FIGHTER_B]["metadata"]["source_robot_name"] is None


def test_sync_locomotion_artifact_writes_separate_registry(tmp_path: Path):
    source = tmp_path / "unitree_g1_velocity.pt"
    _rsl_checkpoint(source)
    export = tmp_path / "policy.onnx"
    export.write_bytes(b"onnx-placeholder")

    record = sync_locomotion_artifact(source, root=tmp_path / "locomotion_bootstrap", exports=[export])

    assert record["not_opponent"] is True
    assert Path(record["checkpoint_path"]).exists()
    assert Path(record["export_paths"][0]).exists()
    assert (tmp_path / "locomotion_bootstrap" / "registry.jsonl").exists()
    assert not (tmp_path / "policy_pool" / "pool.json").exists()


def test_apply_locomotion_warmstart_loads_fake_agent_modules(tmp_path: Path):
    source = tmp_path / "unitree_g1_velocity.pt"
    _rsl_checkpoint(source)
    output = tmp_path / "warmstart.pt"
    create_fight_warmstart(source, output)

    class FakeAgent:
        def __init__(self):
            self.models = {
                FIGHTER_A: {
                    "policy": nn.Sequential(
                        nn.Linear(observation_dim(29), 512),
                        nn.ELU(),
                        nn.Linear(512, 256),
                        nn.ELU(),
                        nn.Linear(256, 128),
                        nn.ELU(),
                        nn.Linear(128, 29),
                    ),
                    "value": nn.Sequential(
                        nn.Linear(observation_dim(29), 512),
                        nn.ELU(),
                        nn.Linear(512, 256),
                        nn.ELU(),
                        nn.Linear(256, 128),
                        nn.ELU(),
                        nn.Linear(128, 1),
                    ),
                }
            }

    agent = FakeAgent()
    loaded = apply_locomotion_warmstart(agent, output)

    assert loaded == [f"{FIGHTER_A}.policy", f"{FIGHTER_A}.value"]
