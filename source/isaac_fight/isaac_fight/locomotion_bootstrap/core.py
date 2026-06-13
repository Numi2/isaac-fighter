# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Warm-start Isaac Fight from upstream Unitree Velocity rsl_rl checkpoints.

The module keeps Unitree locomotion artifacts out of the combat policy pool. It
only turns selected Velocity actor weights into normal skrl-shaped policy/value
state dictionaries that can initialize fight training.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from isaac_fight.assets.robots.unitree import get_unitree_robot_spec
from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
from isaac_fight.tasks.direct.unitree_1v1.observations import observation_dim

LOCOMOTION_WARMSTART_SCHEMA = "isaac_fight.locomotion_warmstart.v1"
DEFAULT_BOOTSTRAP_ROOT = "locomotion_bootstrap"
_HIDDEN_DIMS = (512, 256, 128)


@dataclass(frozen=True)
class VelocityRobotSpec:
    robot_name: str
    task_name: str
    fight_agent: str
    action_dim: int

    @property
    def fight_obs_dim(self) -> int:
        return observation_dim(self.action_dim)


VELOCITY_SPECS: dict[str, VelocityRobotSpec] = {
    "g1_29dof": VelocityRobotSpec(
        robot_name="g1_29dof",
        task_name="Isaac-Velocity-Flat-G1-v0",
        fight_agent=FIGHTER_A,
        action_dim=get_unitree_robot_spec("g1_29dof").action_dim,
    ),
    "h1": VelocityRobotSpec(
        robot_name="h1",
        task_name="Isaac-Velocity-Flat-H1-v0",
        fight_agent=FIGHTER_B,
        action_dim=get_unitree_robot_spec("h1").action_dim,
    ),
}


@dataclass(frozen=True)
class LinearLayerInfo:
    key: str
    bias_key: str | None
    in_features: int
    out_features: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RslRlCheckpointInfo:
    path: str
    source_task: str
    robot_name: str
    action_dim: int
    obs_dim: int
    actor_layers: tuple[LinearLayerInfo, ...]
    critic_layers: tuple[LinearLayerInfo, ...] = ()
    normalizer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["actor_layers"] = [layer.to_json() for layer in self.actor_layers]
        data["critic_layers"] = [layer.to_json() for layer in self.critic_layers]
        return data


@dataclass(frozen=True)
class TransferItem:
    target: str
    source: str | None
    shape: tuple[int, ...]
    status: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WarmstartReport:
    output_path: str
    robot_name: str
    fight_agent: str
    source_checkpoint: str
    transferred: tuple[TransferItem, ...]
    initialized: tuple[TransferItem, ...]

    def print_summary(self) -> None:
        print(f"[INFO] Wrote locomotion warm-start: {self.output_path}", flush=True)
        print(f"[INFO] Source robot: {self.robot_name} -> fight agent(s): {self.fight_agent}", flush=True)
        for item in self.transferred:
            print(f"[INFO] transfer {item.source} -> {item.target} shape={item.shape}", flush=True)
        for item in self.initialized:
            print(f"[INFO] init {item.target} shape={item.shape} reason={item.status}", flush=True)


def inspect_rsl_rl_checkpoint(
    path: str | Path, robot: str | None = None, source_task: str | None = None
) -> RslRlCheckpointInfo:
    """Inspect a Unitree rsl_rl checkpoint without importing Isaac Lab or rsl_rl."""

    checkpoint_path = Path(path)
    checkpoint = _torch_load_cpu(checkpoint_path)
    state_dict = _extract_model_state_dict(checkpoint)
    actor_layers = _extract_linear_layers(state_dict, role="actor")
    if not actor_layers:
        raise ValueError(f"Could not find actor linear layers in rsl_rl checkpoint: {checkpoint_path}")
    critic_layers = _extract_linear_layers(state_dict, role="critic")
    action_dim = int(actor_layers[-1].out_features)
    obs_dim = int(actor_layers[0].in_features)
    robot_name = _resolve_robot_name(robot, source_task, checkpoint_path, action_dim)
    task_name = source_task or VELOCITY_SPECS[robot_name].task_name
    return RslRlCheckpointInfo(
        path=str(checkpoint_path),
        source_task=task_name,
        robot_name=robot_name,
        action_dim=action_dim,
        obs_dim=obs_dim,
        actor_layers=tuple(actor_layers),
        critic_layers=tuple(critic_layers),
        normalizer_shapes=_find_normalizer_shapes(checkpoint),
        metadata=_checkpoint_metadata(checkpoint),
    )


def create_fight_warmstart(
    source_checkpoint: str | Path,
    output_path: str | Path,
    robot: str | None = None,
    source_task: str | None = None,
) -> WarmstartReport:
    """Create a skrl-shaped Isaac Fight checkpoint from one Unitree Velocity checkpoint."""

    source_path = Path(source_checkpoint)
    output = Path(output_path)
    checkpoint = _torch_load_cpu(source_path)
    source_state = _extract_model_state_dict(checkpoint)
    info = inspect_rsl_rl_checkpoint(source_path, robot=robot, source_task=source_task)
    source_spec = VELOCITY_SPECS[info.robot_name]
    if info.action_dim != source_spec.action_dim:
        raise ValueError(
            f"{info.robot_name} expected action_dim={source_spec.action_dim}, source checkpoint has {info.action_dim}"
        )

    fight_agents = _mirror_fight_agents(info.robot_name)
    warmstart: dict[str, Any] = {
        "__metadata__": {
            "schema": LOCOMOTION_WARMSTART_SCHEMA,
            "created_at": time.time(),
            "source_checkpoint": str(source_path),
            "source_task": info.source_task,
            "source_robot": info.robot_name,
            "not_opponent": True,
            "source_sha1": _file_sha1(source_path),
            "normalizer_shapes": {k: list(v) for k, v in info.normalizer_shapes.items()},
            "actor_layers": [layer.to_json() for layer in info.actor_layers],
        }
    }

    transferred: list[TransferItem] = []
    initialized: list[TransferItem] = []
    source_actor_layers = _extract_linear_layers(source_state, role="actor", include_tensors=True)
    source_actor_by_index = {idx: layer for idx, layer in enumerate(source_actor_layers)}

    for agent, target_spec in fight_agents.items():
        obs_dim = target_spec.fight_obs_dim
        action_dim = target_spec.action_dim
        policy, policy_items = _build_skrl_mlp_state(
            input_dim=obs_dim,
            output_dim=action_dim,
            gaussian=True,
            init_log_std=-1.2,
        )
        value, value_items = _build_skrl_mlp_state(input_dim=obs_dim, output_dim=1, gaussian=False)
        initialized.extend(
            TransferItem(target=f"{agent}.policy.{item[0]}", source=None, shape=item[1], status=item[2])
            for item in policy_items
        )
        initialized.extend(
            TransferItem(target=f"{agent}.value.{item[0]}", source=None, shape=item[1], status=item[2])
            for item in value_items
        )

        if target_spec.robot_name == info.robot_name:
            moved, skipped = _copy_compatible_layers(
                source_state=source_state,
                source_layers=source_actor_by_index,
                target_state=policy,
                target_prefix=f"{agent}.policy",
            )
            transferred.extend(moved)
            initialized.extend(skipped)

        warmstart[agent] = {
            "policy": policy,
            "value": value,
            "state_preprocessor": _running_standard_scaler_state(obs_dim),
            "value_preprocessor": _running_standard_scaler_state(1),
            "metadata": {
                "robot_name": target_spec.robot_name,
                "source_robot_name": info.robot_name if target_spec.robot_name == info.robot_name else None,
                "source_task": info.source_task if target_spec.robot_name == info.robot_name else None,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "not_opponent": True,
            },
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warmstart, output)
    return WarmstartReport(
        output_path=str(output),
        robot_name=info.robot_name,
        fight_agent=",".join(
            agent for agent, target_spec in fight_agents.items() if target_spec.robot_name == info.robot_name
        ),
        source_checkpoint=str(source_path),
        transferred=tuple(transferred),
        initialized=tuple(item for item in initialized if item.status != "clean_init"),
    )


def sync_locomotion_artifact(
    source_checkpoint: str | Path,
    root: str | Path = DEFAULT_BOOTSTRAP_ROOT,
    robot: str | None = None,
    source_task: str | None = None,
    exports: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Copy one selected locomotion artifact into ``locomotion_bootstrap`` and append registry metadata."""

    source = Path(source_checkpoint)
    if not source.exists():
        raise FileNotFoundError(source)
    root_path = Path(root)
    checkpoint_dir = root_path / "checkpoints"
    export_dir = root_path / "exports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    info = inspect_rsl_rl_checkpoint(source, robot=robot, source_task=source_task)
    digest = _file_sha1(source)[:12]
    checkpoint_target = checkpoint_dir / f"{info.robot_name}_{digest}_{source.name}"
    if source.resolve() != checkpoint_target.resolve():
        shutil.copy2(source, checkpoint_target)

    export_targets: list[str] = []
    for export in exports:
        export_path = Path(export)
        if not export_path.exists():
            raise FileNotFoundError(export_path)
        target = export_dir / f"{info.robot_name}_{digest}_{export_path.name}"
        if export_path.resolve() != target.resolve():
            shutil.copy2(export_path, target)
        export_targets.append(str(target))

    record = {
        "schema": "isaac_fight.locomotion_artifact.v1",
        "created_at": time.time(),
        "kind": "unitree_velocity_checkpoint",
        "robot": info.robot_name,
        "source_task": info.source_task,
        "source_path": str(source),
        "checkpoint_path": str(checkpoint_target),
        "export_paths": export_targets,
        "obs_dim": info.obs_dim,
        "action_dim": info.action_dim,
        "actor_layers": [layer.to_json() for layer in info.actor_layers],
        "normalizer_shapes": {k: list(v) for k, v in info.normalizer_shapes.items()},
        "source_sha1": _file_sha1(source),
        "not_opponent": True,
    }
    registry = root_path / "registry.jsonl"
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def sync_motion_prior_artifact(
    motion_file: str | Path,
    root: str | Path = DEFAULT_BOOTSTRAP_ROOT,
    robot: str = "g1_29dof",
    source_task: str = "Unitree-G1-29dof-Mimic",
    kind: str = "unitree_g1_mimic_motion",
) -> dict[str, Any]:
    """Copy one AMP/mimic motion-prior artifact into ``locomotion_bootstrap`` and append registry metadata."""

    source = Path(motion_file)
    if not source.exists():
        raise FileNotFoundError(source)
    root_path = Path(root)
    motion_dir = root_path / "motion_priors"
    motion_dir.mkdir(parents=True, exist_ok=True)
    digest = _file_sha1(source)[:12]
    target = motion_dir / f"{robot}_{digest}_{source.name}"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    record = {
        "schema": "isaac_fight.motion_prior_artifact.v1",
        "created_at": time.time(),
        "kind": kind,
        "robot": _normalize_robot(robot),
        "source_task": source_task,
        "source_path": str(source),
        "artifact_path": str(target),
        "source_sha1": _file_sha1(source),
        "bootstrap_use": "amp_or_mimic_pretraining",
        "not_opponent": True,
    }
    registry = root_path / "motion_priors.jsonl"
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def is_locomotion_warmstart_checkpoint(path: str | Path) -> bool:
    try:
        checkpoint = _torch_load_cpu(Path(path))
    except Exception:
        return False
    metadata = checkpoint.get("__metadata__") if isinstance(checkpoint, dict) else None
    return isinstance(metadata, dict) and metadata.get("schema") == LOCOMOTION_WARMSTART_SCHEMA


def apply_locomotion_warmstart(agent: Any, path: str | Path) -> list[str]:
    """Best-effort loader for generated warm-starts against live skrl agents.

    skrl checkpoint loading is strict about model architecture. This helper keeps
    ``scripts/skrl/train.py --checkpoint`` usable for generated warm-starts by
    loading only matching policy/value modules and leaving optimizer state fresh.
    """

    checkpoint = _torch_load_cpu(Path(path))
    metadata = checkpoint.get("__metadata__") if isinstance(checkpoint, dict) else None
    if not isinstance(metadata, dict) or metadata.get("schema") != LOCOMOTION_WARMSTART_SCHEMA:
        return []

    loaded: list[str] = []
    for agent_id in (FIGHTER_A, FIGHTER_B):
        payload = checkpoint.get(agent_id)
        if not isinstance(payload, dict):
            continue
        for model_name in ("policy", "value"):
            state_dict = payload.get(model_name)
            if not isinstance(state_dict, dict):
                continue
            module = _find_agent_module(agent, agent_id, model_name)
            if module is None or not hasattr(module, "load_state_dict"):
                continue
            if _load_shape_compatible_state_dict(module, state_dict):
                loaded.append(f"{agent_id}.{model_name}")
    return loaded


def _mirror_fight_agents(robot_name: str) -> dict[str, VelocityRobotSpec]:
    spec = VELOCITY_SPECS[robot_name]
    return {FIGHTER_A: spec, FIGHTER_B: spec}


def _load_shape_compatible_state_dict(module: Any, state_dict: dict[str, torch.Tensor]) -> bool:
    try:
        target_state = module.state_dict()
    except Exception:
        module.load_state_dict(state_dict, strict=False)
        return True
    compatible = {}
    for key, value in state_dict.items():
        for target_key in (key, key.removeprefix("net_container.")):
            target = target_state.get(target_key)
            if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor):
                try:
                    if tuple(value.shape) == tuple(target.shape):
                        compatible[target_key] = value
                        break
                except RuntimeError:
                    continue
    if not compatible:
        return False
    module.load_state_dict(compatible, strict=False)
    return True


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "actor_critic_state_dict", "state_dict", "model", "actor_critic"):
            value = checkpoint.get(key)
            if _is_tensor_state_dict(value):
                return value
        if _is_tensor_state_dict(checkpoint):
            return checkpoint
        for value in checkpoint.values():
            if _is_tensor_state_dict(value):
                return value
    raise ValueError("Could not find a tensor state dict in the checkpoint")


def _is_tensor_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values())


def _extract_linear_layers(
    state_dict: dict[str, torch.Tensor],
    role: str,
    include_tensors: bool = False,
) -> list[Any]:
    weight_keys = [
        key
        for key, value in state_dict.items()
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2
    ]
    role_keys = _filter_role_keys(weight_keys, role)
    layers = []
    for key in sorted(role_keys, key=_natural_sort_key):
        weight = state_dict[key]
        prefix = key[: -len(".weight")]
        bias_key = f"{prefix}.bias" if isinstance(state_dict.get(f"{prefix}.bias"), torch.Tensor) else None
        info = LinearLayerInfo(
            key=key,
            bias_key=bias_key,
            in_features=int(weight.shape[1]),
            out_features=int(weight.shape[0]),
        )
        if include_tensors:
            layers.append((info, weight, state_dict.get(bias_key) if bias_key else None))
        else:
            layers.append(info)
    return layers


def _filter_role_keys(weight_keys: list[str], role: str) -> list[str]:
    lowered = {key: key.lower() for key in weight_keys}
    if role == "actor":
        preferred = [
            key
            for key, low in lowered.items()
            if any(token in low for token in ("actor", "policy", "mean"))
            and not any(token in low for token in ("critic", "value"))
        ]
        if preferred:
            return preferred
        return [key for key, low in lowered.items() if not any(token in low for token in ("critic", "value"))]
    preferred = [key for key, low in lowered.items() if any(token in low for token in ("critic", "value"))]
    return preferred


def _natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _resolve_robot_name(robot: str | None, source_task: str | None, path: Path, action_dim: int) -> str:
    if robot:
        return _normalize_robot(robot)
    text = f"{source_task or ''} {path}".lower()
    if "g1" in text and "29" in text:
        return "g1_29dof"
    if re.search(r"(^|[^a-z0-9])h1([^a-z0-9]|$)", text):
        return "h1"
    matches = [name for name, spec in VELOCITY_SPECS.items() if spec.action_dim == action_dim]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not infer Unitree robot from action_dim={action_dim}; pass robot explicitly")


def _normalize_robot(robot: str) -> str:
    normalized = robot.lower().replace("-", "_")
    aliases = {"g1": "g1_29dof", "g1_29": "g1_29dof", "g1_29_dof": "g1_29dof", "g1_29dof": "g1_29dof", "h1": "h1"}
    if normalized not in aliases:
        raise KeyError(f"Unsupported Velocity bootstrap robot: {robot}. Supported: {', '.join(VELOCITY_SPECS)}")
    return aliases[normalized]


def _checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("iter", "iteration", "infos"):
        value = checkpoint.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, dict):
            metadata[key] = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
    return metadata


def _find_normalizer_shapes(value: Any) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    def visit(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, child in obj.items():
                key_text = str(key)
                next_prefix = f"{prefix}.{key_text}" if prefix else key_text
                if isinstance(child, torch.Tensor) and any(
                    token in next_prefix.lower() for token in ("normal", "obs", "rms")
                ):
                    if any(token in key_text.lower() for token in ("mean", "var", "variance", "std")):
                        shapes[next_prefix] = tuple(int(dim) for dim in child.shape)
                elif isinstance(child, dict):
                    visit(child, next_prefix)

    visit(value, "")
    return shapes


def _build_skrl_mlp_state(
    input_dim: int,
    output_dim: int,
    gaussian: bool,
    init_log_std: float = -1.2,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], str]]]:
    dims = (int(input_dim), *_HIDDEN_DIMS, int(output_dim))
    state: dict[str, torch.Tensor] = {}
    report: list[tuple[str, tuple[int, ...], str]] = []
    for layer_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        key_index = layer_index * 2
        weight = torch.empty(out_dim, in_dim, dtype=torch.float32)
        gain = math.sqrt(2.0) if layer_index < len(dims) - 2 else 0.01
        torch.nn.init.orthogonal_(weight, gain=gain)
        bias = torch.zeros(out_dim, dtype=torch.float32)
        weight_key = f"net_container.{key_index}.weight"
        bias_key = f"net_container.{key_index}.bias"
        state[weight_key] = weight
        state[bias_key] = bias
        report.append((weight_key, tuple(weight.shape), "clean_init"))
        report.append((bias_key, tuple(bias.shape), "clean_init"))
    if gaussian:
        state["log_std_parameter"] = torch.full((output_dim,), float(init_log_std), dtype=torch.float32)
        report.append(("log_std_parameter", (int(output_dim),), "clean_init"))
    return state, report


def _copy_compatible_layers(
    source_state: dict[str, torch.Tensor],
    source_layers: dict[int, Any],
    target_state: dict[str, torch.Tensor],
    target_prefix: str,
) -> tuple[list[TransferItem], list[TransferItem]]:
    transferred: list[TransferItem] = []
    skipped: list[TransferItem] = []
    target_weight_keys = sorted(
        [key for key, value in target_state.items() if key.endswith(".weight") and value.ndim == 2],
        key=_natural_sort_key,
    )
    for index, target_weight_key in enumerate(target_weight_keys):
        source = source_layers.get(index)
        if source is None:
            continue
        source_info, source_weight, source_bias = source
        target_weight = target_state[target_weight_key]
        target_bias_key = target_weight_key[: -len(".weight")] + ".bias"
        if tuple(source_weight.shape) != tuple(target_weight.shape):
            skipped.append(
                TransferItem(
                    target=f"{target_prefix}.{target_weight_key}",
                    source=source_info.key,
                    shape=tuple(int(dim) for dim in target_weight.shape),
                    status="shape_mismatch_reinitialized",
                )
            )
            continue
        target_state[target_weight_key] = source_weight.detach().float().clone()
        transferred.append(
            TransferItem(
                target=f"{target_prefix}.{target_weight_key}",
                source=source_info.key,
                shape=tuple(int(dim) for dim in source_weight.shape),
                status="copied",
            )
        )
        if (
            target_bias_key in target_state
            and isinstance(source_bias, torch.Tensor)
            and tuple(source_bias.shape) == tuple(target_state[target_bias_key].shape)
        ):
            target_state[target_bias_key] = source_bias.detach().float().clone()
            transferred.append(
                TransferItem(
                    target=f"{target_prefix}.{target_bias_key}",
                    source=source_info.bias_key,
                    shape=tuple(int(dim) for dim in source_bias.shape),
                    status="copied",
                )
            )
    return transferred, skipped


def _running_standard_scaler_state(size: int) -> dict[str, torch.Tensor]:
    return {
        "running_mean": torch.zeros(int(size), dtype=torch.float32),
        "running_variance": torch.ones(int(size), dtype=torch.float32),
        "current_count": torch.tensor(1.0, dtype=torch.float32),
    }


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_agent_module(agent: Any, agent_id: str, model_name: str) -> Any | None:
    direct_agents = getattr(agent, "agents", None)
    if isinstance(direct_agents, dict) and agent_id in direct_agents:
        found = _find_agent_module(direct_agents[agent_id], agent_id, model_name)
        if found is not None:
            return found

    for attr in ("models", "_models", "modules", "_modules"):
        container = getattr(agent, attr, None)
        found = _find_in_container(container, agent_id, model_name, depth=0)
        if found is not None:
            return found
    return None


def _find_in_container(container: Any, agent_id: str, model_name: str, depth: int) -> Any | None:
    if depth > 4:
        return None
    if isinstance(container, dict):
        direct_patterns = (
            (agent_id, model_name),
            (model_name, agent_id),
            (f"{agent_id}_{model_name}",),
            (f"{agent_id}/{model_name}",),
            (f"{model_name}/{agent_id}",),
        )
        for pattern in direct_patterns:
            value = container
            for key in pattern:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None and hasattr(value, "load_state_dict"):
                return value
        for key, value in container.items():
            key_text = str(key)
            if key_text in {agent_id, model_name} or agent_id in key_text or model_name in key_text:
                found = _find_in_container(value, agent_id, model_name, depth + 1)
                if found is not None:
                    return found
    return container if hasattr(container, "load_state_dict") and depth > 0 else None
