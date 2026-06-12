from __future__ import annotations

from isaac_fight.assets.robots.unitree import get_unitree_robot_spec, list_supported_unitree_robots


def test_unitree_action_dimensions():
    assert "g1_29dof" in list_supported_unitree_robots()
    assert get_unitree_robot_spec("g1_29dof").action_dim == 29
    assert get_unitree_robot_spec("h1").action_dim == 19
