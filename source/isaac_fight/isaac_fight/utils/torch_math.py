# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch math utilities that avoid importing Isaac Lab in unit tests."""

from __future__ import annotations

import math

import torch


def normalize(x: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    # Isaac Lab root quaternions use wxyz ordering.
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack((torch.cos(half), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(half)), dim=-1)


def rotate_yaw_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(quat_from_yaw(yaw_from_quat(q)), v)


def heading_error_to_target(root_quat_w: torch.Tensor, rel_pos_w: torch.Tensor) -> torch.Tensor:
    yaw = yaw_from_quat(root_quat_w)
    target_yaw = torch.atan2(rel_pos_w[..., 1], rel_pos_w[..., 0])
    return wrap_to_pi(target_yaw - yaw)


def exp_reward(error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.square(error) / max(std * std, 1.0e-8))
