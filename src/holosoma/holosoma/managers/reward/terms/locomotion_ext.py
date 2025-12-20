
from __future__ import annotations

from typing import TYPE_CHECKING
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def pose(
    env: LeggedRobotLocomotionManager,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32).unsqueeze(0)
    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos_base)
    if env.walk_dof_pos_base is not None:
        walk_error = torch.square(qpos - env.walk_dof_pos_base)
    else:
        walk_error = pose_error

    ##
    command_tensor = getattr(env.command_manager, "commands")
    if command_tensor is not None:
        stand_mask = torch.logical_and(
                torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
                torch.abs(command_tensor[:, 2]) < 0.01,
            )

        if stand_mask.any():
            walk_mask = torch.logical_not(stand_mask)
            max_weights = max(pose_weights)
            pose_error[stand_mask, :] *= max_weights
            pose_error[walk_mask, :] = walk_error[walk_mask, :] * weights
            weighted_error = pose_error
        else:
            weighted_error = walk_error * weights

    return torch.sum(weighted_error, dim=1)


def penalty_feet_contact_forces(env, force_threshold: float = 450) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    forces = torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2])
    forces = torch.clamp_min(forces - force_threshold, min=0)
    return torch.sum(forces, dim=1)


def penalty_feet_contact_forces_v1(env, force_threshold: float = 450, max_force: float = 400) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    forces = torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2])
    _reward = torch.clamp(forces - force_threshold, min=0, max=max_force)
    _reward = torch.max(_reward, dim=1)[0]
    return _reward
