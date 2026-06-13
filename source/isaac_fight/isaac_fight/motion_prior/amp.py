"""AMP-style motion-prior feature and discriminator helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

AMP_FEATURE_SCHEMA = (
    "joint_pos_rel",
    "joint_vel_scaled",
    "root_lin_vel_b_scaled",
    "root_ang_vel_b_scaled",
    "projected_gravity_b",
    "root_height_ratio",
    "support_quality",
    "up_z",
)


class MotionPriorDiscriminator(nn.Module):
    """MLP discriminator used for AMP-style motion-prior rewards."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ELU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def amp_feature_dim(action_dim: int) -> int:
    return int(action_dim) * 2 + 12


def read_motion_file(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npz":
        import numpy as np

        with np.load(path, allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"unsupported motion artifact type: {type(raw)!r}")


def build_reference_amp_features(
    raw: dict[str, Any],
    target_joint_names: tuple[str, ...] | list[str],
    *,
    default_base_height: float = 0.8,
    joint_velocity_scale: float = 0.05,
    base_linear_velocity_scale: float = 0.5,
    base_angular_velocity_scale: float = 0.2,
    min_joint_name_coverage: float = 0.80,
    allow_unnamed_dim_match: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    joint_names = motion_joint_names(raw)
    joint_pos = motion_tensor(
        raw,
        ("joint_pos_rel", "dof_pos_rel", "joint_position_rel", "joint_positions_rel"),
        device=device,
    )
    if joint_pos is None:
        joint_pos = motion_tensor(raw, ("joint_pos", "joint_positions", "dof_pos", "qpos", "target_joint_pos"), device=device)
    if joint_pos is None:
        raise ValueError("motion artifact has no joint position tensor")
    joint_pos = adapt_motion_joints(
        joint_pos,
        joint_names,
        target_joint_names,
        min_joint_name_coverage=min_joint_name_coverage,
        allow_unnamed_dim_match=allow_unnamed_dim_match,
    )
    frame_count = int(joint_pos.shape[0])
    action_dim = int(joint_pos.shape[-1])

    joint_vel = motion_tensor(raw, ("joint_vel", "joint_velocities", "dof_vel", "qvel", "target_joint_vel"), device=device)
    if joint_vel is None:
        joint_vel = torch.zeros(frame_count, action_dim, device=device)
    else:
        joint_vel = _match_rows(
            adapt_motion_joints(
                joint_vel,
                joint_names,
                target_joint_names,
                min_joint_name_coverage=min_joint_name_coverage,
                allow_unnamed_dim_match=allow_unnamed_dim_match,
            ),
            frame_count,
        )

    root_lin_vel = motion_tensor(
        raw,
        ("root_lin_vel_b", "root_lin_vel", "base_lin_vel", "base_linear_velocity", "root_velocity"),
        device=device,
    )
    root_lin_vel = _motion_or_zeros(root_lin_vel, frame_count, 3, device)

    root_ang_vel = motion_tensor(
        raw,
        ("root_ang_vel_b", "root_ang_vel", "base_ang_vel", "base_angular_velocity"),
        device=device,
    )
    root_ang_vel = _motion_or_zeros(root_ang_vel, frame_count, 3, device)

    projected_gravity = motion_tensor(raw, ("projected_gravity_b", "projected_gravity"), device=device)
    if projected_gravity is None:
        projected_gravity = torch.zeros(frame_count, 3, device=device)
        projected_gravity[:, 2] = -1.0
    else:
        projected_gravity = _match_rows(projected_gravity[:, :3], frame_count)

    root_height = motion_tensor(raw, ("root_height", "base_height", "pelvis_height"), device=device)
    if root_height is not None:
        root_height = _match_rows(root_height[:, :1], frame_count)
    else:
        root_pos = motion_tensor(raw, ("root_pos", "root_position", "base_pos", "base_position"), device=device)
        root_height = _match_rows(root_pos[:, 2:3], frame_count) if root_pos is not None and root_pos.shape[-1] >= 3 else None
    if root_height is None:
        root_height = torch.full((frame_count, 1), float(default_base_height), device=device)
    root_height_ratio = root_height / max(float(default_base_height), 1.0e-6)

    support_quality = motion_tensor(raw, ("support_quality", "feet_ground_support"), device=device)
    support_quality = (
        _match_rows(support_quality[:, :1], frame_count)
        if support_quality is not None
        else torch.ones(frame_count, 1, device=device)
    )

    up_z = motion_tensor(raw, ("up_z", "root_up_z", "upright"), device=device)
    up_z = _match_rows(up_z[:, :1], frame_count) if up_z is not None else torch.ones(frame_count, 1, device=device)

    features = torch.cat(
        (
            torch.clamp(joint_pos, -5.0, 5.0),
            torch.clamp(joint_vel * float(joint_velocity_scale), -5.0, 5.0),
            torch.clamp(root_lin_vel[:, :3] * float(base_linear_velocity_scale), -5.0, 5.0),
            torch.clamp(root_ang_vel[:, :3] * float(base_angular_velocity_scale), -5.0, 5.0),
            torch.clamp(projected_gravity[:, :3], -1.0, 1.0),
            torch.clamp(root_height_ratio, 0.0, 3.0),
            torch.clamp(support_quality, 0.0, 1.0),
            torch.clamp(up_z, -1.0, 1.0),
        ),
        dim=-1,
    )
    expected = amp_feature_dim(action_dim)
    if features.shape[-1] != expected:
        raise RuntimeError(f"AMP feature dim {features.shape[-1]} != expected {expected}")
    return features


def load_amp_feature_files(paths: list[str] | tuple[str, ...], device: str | torch.device = "cpu") -> torch.Tensor:
    features: list[torch.Tensor] = []
    for path_value in paths:
        path = Path(path_value).expanduser()
        if path.suffix.lower() == ".npz":
            raw = read_motion_file(path)
            value = motion_value(raw, ("amp_features", "features", "rollout_features"))
            if value is None:
                raise ValueError(f"{path} has no amp_features/features tensor")
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            raw = torch.load(path, map_location=device)
            if isinstance(raw, dict):
                value = motion_value(raw, ("amp_features", "features", "rollout_features"))
                if value is None:
                    raise ValueError(f"{path} has no amp_features/features tensor")
                tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            else:
                tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        features.append(tensor)
    if not features:
        raise ValueError("no feature files supplied")
    return torch.cat(features, dim=0)


def motion_tensor(raw: dict[str, Any], keys: tuple[str, ...], device: str | torch.device = "cpu") -> torch.Tensor | None:
    value = motion_value(raw, keys)
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        return None
    return tensor


def motion_value(raw: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in raw:
            return raw[key]
    for container_key in ("motion", "motions", "data", "reference", "demo"):
        nested = raw.get(container_key)
        if hasattr(nested, "shape") and getattr(nested, "shape", ()) == ():
            try:
                nested = nested.item()
            except Exception:
                nested = None
        if isinstance(nested, dict):
            value = motion_value(nested, keys)
            if value is not None:
                return value
    return None


def motion_joint_names(raw: dict[str, Any]) -> list[str]:
    value = motion_value(raw, ("joint_names", "dof_names", "controlled_joint_names"))
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    names: list[str] = []
    for item in value:
        names.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return names


def adapt_motion_joints(
    tensor: torch.Tensor,
    source_joint_names: list[str],
    target_joint_names: tuple[str, ...] | list[str],
    *,
    min_joint_name_coverage: float = 0.80,
    allow_unnamed_dim_match: bool = True,
) -> torch.Tensor:
    action_dim = len(target_joint_names)
    if source_joint_names:
        names = list(source_joint_names)
        if len(names) < tensor.shape[-1]:
            names = [*names, *(f"__unnamed_{idx}" for idx in range(len(names), tensor.shape[-1]))]
        elif len(names) > tensor.shape[-1]:
            names = names[: tensor.shape[-1]]
        index_by_name = {name: idx for idx, name in enumerate(names)}
        out = torch.zeros(tensor.shape[0], action_dim, device=tensor.device, dtype=tensor.dtype)
        matched = 0
        missing: list[str] = []
        for out_idx, name in enumerate(target_joint_names):
            source_idx = index_by_name.get(name)
            if source_idx is not None and source_idx < tensor.shape[-1]:
                out[:, out_idx] = tensor[:, source_idx]
                matched += 1
            else:
                missing.append(str(name))
        coverage = matched / max(action_dim, 1)
        required = max(0.0, min(1.0, float(min_joint_name_coverage)))
        if coverage < required:
            raise ValueError(
                f"motion joint-name coverage {coverage:.2f} below required {required:.2f}; "
                f"missing examples: {missing[:8]}"
            )
        return out
    if tensor.shape[-1] == action_dim and allow_unnamed_dim_match:
        return tensor
    raise ValueError(
        f"motion joint tensor width {tensor.shape[-1]} does not match target action_dim={action_dim} "
        "and no joint_names/dof_names were provided for safe mapping"
    )


def _match_rows(tensor: torch.Tensor, frame_count: int) -> torch.Tensor:
    if tensor.shape[0] == frame_count:
        return tensor
    index = torch.arange(frame_count, device=tensor.device) % int(tensor.shape[0])
    return tensor.index_select(0, index)


def _motion_or_zeros(
    tensor: torch.Tensor | None,
    frame_count: int,
    width: int,
    device: str | torch.device,
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(frame_count, width, device=device)
    if tensor.shape[-1] < width:
        pad = torch.zeros(tensor.shape[0], width - tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, pad), dim=-1)
    return _match_rows(tensor[:, :width], frame_count)
