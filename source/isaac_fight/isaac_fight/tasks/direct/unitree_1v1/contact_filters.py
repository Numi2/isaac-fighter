# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact force filters for combat attribution."""

from __future__ import annotations

from collections.abc import Sequence

import torch

LOWER_BODY_GROUND_CONTACT_TOKENS = ("foot", "ankle", "toe", "sole")


def max_combat_contact_force(
    forces: torch.Tensor,
    *,
    body_names: Sequence[str] = (),
    include_lower_body_contacts: torch.Tensor | bool | None = None,
    lower_body_tokens: Sequence[str] = LOWER_BODY_GROUND_CONTACT_TOKENS,
) -> torch.Tensor:
    """Return max body contact force while suppressing far-field walking contact.

    Lower-body contacts are useful fight signals only when the fighter is in the
    opponent engagement window. Away from the opponent, feet and ankles mostly
    describe normal ground reaction forces.
    """

    all_body_force = torch.linalg.norm(forces, dim=-1).amax(dim=-1)
    if not body_names or forces.shape[1] != len(body_names):
        return all_body_force

    lower_tokens = tuple(token.lower() for token in lower_body_tokens)
    upper_body_mask = [
        not any(token in body_name.lower() for token in lower_tokens)
        for body_name in body_names
    ]
    if not any(upper_body_mask):
        return all_body_force

    upper_forces = forces[:, torch.tensor(upper_body_mask, dtype=torch.bool, device=forces.device)]
    upper_body_force = torch.linalg.norm(upper_forces, dim=-1).amax(dim=-1)
    if include_lower_body_contacts is None:
        return upper_body_force

    include = torch.as_tensor(include_lower_body_contacts, dtype=torch.bool, device=forces.device)
    if include.ndim == 0:
        include = include.expand_as(all_body_force)
    return torch.where(include, all_body_force, upper_body_force)
