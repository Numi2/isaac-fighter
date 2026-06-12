from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source" / "isaac_fight" / "isaac_fight" / "tasks" / "direct" / "unitree_1v1"


def test_task_registration_static_contract():
    text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "GhostFighter-Unitree-1v1-Direct-v0" in text
    assert "skrl_ippo_cfg_entry_point" in text
    assert "skrl_mappo_cfg_entry_point" in text
    assert "Direct" in text


def test_environment_contains_required_rule_hooks():
    text = (TASK_DIR / "unitree_1v1_env.py").read_text(encoding="utf-8")
    for hook in ["_get_observations", "_get_rewards", "_get_dones", "reset_agent_state", "_write_replay_step"]:
        assert f"def {hook}" in text
    for concept in ["knockdown", "out_of_bounds", "winner", "contact_force", "energy"]:
        assert concept in text


def test_reward_terms_cover_requested_combat_terms():
    text = (TASK_DIR / "reward_terms.py").read_text(encoding="utf-8")
    for term in [
        "upright_stability",
        "balance_recovery",
        "controlled_approach",
        "useful_contact",
        "destabilizing_impact",
        "opponent_knockdown",
        "final_win",
        "spin_without_contact",
        "uncontrolled_collision",
    ]:
        assert term in text
