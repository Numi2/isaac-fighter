#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Train an AMP-style discriminator from reference G1 motion and policy rollout features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "isaac_fight"))

import torch
import torch.nn.functional as F

from isaac_fight.assets.robots.unitree import get_unitree_robot_spec
from isaac_fight.motion_prior.amp import (
    AMP_FEATURE_SCHEMA,
    MotionPriorDiscriminator,
    amp_feature_dim,
    build_reference_amp_features,
    load_amp_feature_files,
    read_motion_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_prior_artifact", required=True, help="Reference motion .npz/.pt artifact.")
    parser.add_argument(
        "--negative_features",
        nargs="*",
        default=(),
        help="Policy rollout AMP feature .pt/.npz files from export_amp_rollout_features.py.",
    )
    parser.add_argument("--output", required=True, help="Output discriminator checkpoint path.")
    parser.add_argument("--robot", default="g1_29dof")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=(256, 256))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--max_reference_samples", type=int, default=200_000)
    parser.add_argument("--max_negative_samples", type=int, default=200_000)
    parser.add_argument("--synthetic_negative_ratio", type=float, default=1.0)
    parser.add_argument("--synthetic_noise", type=float, default=0.45)
    parser.add_argument("--min_joint_name_coverage", type=float, default=0.80)
    parser.add_argument(
        "--disallow_unnamed_dim_match",
        action="store_true",
        default=False,
        help="Reject motion artifacts without joint names even when tensor width matches the robot action dimension.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torchscript_output", default="", help="Optional TorchScript discriminator output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    spec = get_unitree_robot_spec(args.robot)
    raw = read_motion_file(args.motion_prior_artifact)
    reference = build_reference_amp_features(
        raw,
        spec.controlled_joint_names,
        default_base_height=spec.default_base_height,
        min_joint_name_coverage=args.min_joint_name_coverage,
        allow_unnamed_dim_match=not args.disallow_unnamed_dim_match,
        device=device,
    )
    reference = _subsample(reference, args.max_reference_samples)
    if args.negative_features:
        negative = load_amp_feature_files(tuple(args.negative_features), device=device)
        if negative.shape[-1] != reference.shape[-1]:
            raise RuntimeError(f"negative feature dim {negative.shape[-1]} != reference dim {reference.shape[-1]}")
        negative = _subsample(negative, args.max_negative_samples)
    else:
        count = max(1, int(reference.shape[0] * max(float(args.synthetic_negative_ratio), 0.0)))
        negative = _synthetic_negatives(reference, count=count, action_dim=spec.action_dim, noise=float(args.synthetic_noise))

    features = torch.cat((reference, negative), dim=0)
    labels = torch.cat(
        (
            torch.ones(reference.shape[0], device=device),
            torch.zeros(negative.shape[0], device=device),
        ),
        dim=0,
    )
    input_dim = int(features.shape[-1])
    expected_dim = amp_feature_dim(spec.action_dim)
    if input_dim != expected_dim:
        raise RuntimeError(f"AMP feature dim {input_dim} != expected {expected_dim} for {args.robot}")

    model = MotionPriorDiscriminator(input_dim, tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for epoch in range(1, int(args.epochs) + 1):
        loss, accuracy = _train_epoch(model, optimizer, features, labels, batch_size=int(args.batch_size))
        if epoch == 1 or epoch == int(args.epochs) or epoch % 5 == 0:
            print(f"[INFO] epoch={epoch:04d} loss={loss:.5f} acc={accuracy:.3f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "isaac_fight.amp_discriminator.v1",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": input_dim,
        "hidden_dims": tuple(int(dim) for dim in args.hidden_dims),
        "feature_schema": AMP_FEATURE_SCHEMA,
        "robot": args.robot,
        "reference_samples": int(reference.shape[0]),
        "negative_samples": int(negative.shape[0]),
        "motion_prior_artifact": str(Path(args.motion_prior_artifact).expanduser()),
        "negative_feature_files": [str(Path(path).expanduser()) for path in args.negative_features],
        "min_joint_name_coverage": float(args.min_joint_name_coverage),
        "allow_unnamed_dim_match": not bool(args.disallow_unnamed_dim_match),
    }
    torch.save(payload, output)
    if args.torchscript_output:
        scripted = torch.jit.trace(model.eval().cpu(), torch.zeros(1, input_dim))
        ts_path = Path(args.torchscript_output)
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(ts_path))
    print(json.dumps({"output": str(output.resolve()), "input_dim": input_dim, "schema": payload["schema"]}))


def _train_epoch(
    model: MotionPriorDiscriminator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    model.train()
    order = torch.randperm(features.shape[0], device=features.device)
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for start in range(0, order.numel(), max(1, batch_size)):
        idx = order[start : start + batch_size]
        x = features.index_select(0, idx)
        y = labels.index_select(0, idx)
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.sigmoid(logits) >= 0.5
            total_correct += float((pred == y.bool()).float().sum().item())
            total += int(idx.numel())
            total_loss += float(loss.item()) * int(idx.numel())
    return total_loss / max(total, 1), total_correct / max(total, 1)


def _subsample(tensor: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0 or tensor.shape[0] <= max_samples:
        return tensor
    return tensor.index_select(0, torch.randperm(tensor.shape[0], device=tensor.device)[:max_samples])


def _synthetic_negatives(reference: torch.Tensor, *, count: int, action_dim: int, noise: float) -> torch.Tensor:
    index = torch.randint(reference.shape[0], (count,), device=reference.device)
    negative = reference.index_select(0, index).clone()
    negative[:, :action_dim] += noise * torch.randn_like(negative[:, :action_dim])
    negative[:, action_dim : 2 * action_dim] += 2.0 * noise * torch.randn_like(negative[:, action_dim : 2 * action_dim])
    extra_start = 2 * action_dim
    projected_gravity_start = extra_start + 6
    height_idx = extra_start + 9
    support_idx = extra_start + 10
    up_idx = extra_start + 11
    random_gravity = torch.randn(count, 3, device=reference.device)
    random_gravity = random_gravity / torch.clamp(torch.linalg.norm(random_gravity, dim=-1, keepdim=True), min=1.0e-6)
    negative[:, projected_gravity_start : projected_gravity_start + 3] = random_gravity
    negative[:, height_idx] = torch.rand(count, device=reference.device) * 1.6
    negative[:, support_idx] = torch.rand(count, device=reference.device) * 0.45
    negative[:, up_idx] = torch.rand(count, device=reference.device) * 1.4 - 0.35
    return negative


if __name__ == "__main__":
    main()
