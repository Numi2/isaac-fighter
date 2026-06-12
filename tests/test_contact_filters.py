from __future__ import annotations

import torch
from isaac_fight.tasks.direct.unitree_1v1.contact_filters import max_combat_contact_force


def test_lower_body_contact_is_available_only_inside_attack_window():
    forces = torch.zeros(2, 3, 3)
    forces[:, 0, 0] = 10.0
    forces[:, 1, 0] = 90.0
    forces[:, 2, 0] = 50.0

    contact_force = max_combat_contact_force(
        forces,
        body_names=("torso_link", "left_ankle_roll_link", "right_sole_link"),
        include_lower_body_contacts=torch.tensor([False, True]),
    )

    assert torch.allclose(contact_force, torch.tensor([10.0, 90.0]))


def test_lower_body_contact_filter_falls_back_when_names_do_not_match_forces():
    forces = torch.zeros(1, 2, 3)
    forces[0, 0, 0] = 4.0
    forces[0, 1, 0] = 7.0

    contact_force = max_combat_contact_force(
        forces,
        body_names=("torso_link",),
        include_lower_body_contacts=False,
    )

    assert torch.allclose(contact_force, torch.tensor([7.0]))
