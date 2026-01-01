
from __future__ import annotations

from typing import TYPE_CHECKING
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager

def penalty_pose_maxoffset(
    env: LeggedRobotLocomotionManager,
    joint_names: list[str],
    max_offset: list[float],
) -> torch.Tensor:
    # Get current joint positions
    joint_ids = []
    for name in joint_names:
        joint_ids.append(env.dof_names.index(name))

    pose_error = torch.abs(env.simulator.dof_pos[:, joint_ids] - env.default_dof_pos_base[:, joint_ids])
    max_offset = torch.tensor(max_offset, device = pose_error.device)[None, :]
    offset_error = torch.clamp_min(pose_error - max_offset, 0)
    offset_error = torch.square(offset_error)

    return torch.sum(offset_error, dim=1)

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

def reward_shoulder_gait(env,
    swing_range: float = 0.3,
    swing_sigma: float = 0.08,
    shoulde_joint_names: list[str] = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
    hip_joint_names: list[str] = ["left_hip_pitch_joint", "right_hip_pitch_joint"]) -> torch.Tensor:

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    swing_target = torch.abs(torch.cos(gait_state.phase + torch.pi) * swing_range)

    shoulde_ids = [env.dof_names.index(name) for name in shoulde_joint_names]
    hip_ids = [env.dof_names.index(name) for name in hip_joint_names]
    #
    qpos = env.simulator.dof_pos
    default_qpos = env.default_dof_pos

    hip_sign = torch.sign(qpos[:, hip_ids] - default_qpos[:, shoulde_ids]) * -1
    swing_target *= hip_sign

    command_tensor = getattr(env.manager, "commands", None) if hasattr(env, "manager") else None
    if command_tensor is not None:
        stand_mask = torch.logical_and(
            torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
            torch.abs(command_tensor[:, 2]) < 0.01,
        )
        if stand_mask.any():
            swing_target[stand_mask]=0

    shoulder_pos = qpos[:, shoulde_ids] - default_qpos[:, shoulde_ids]

    error = torch.sum(torch.square(shoulder_pos - swing_target), dim=-1)
    return torch.exp(-error / swing_sigma)


def penalty_shoulder_gait_signwithlinevel(env,
    swing_range: float = 0.25,
    cent_pos: float = 0.15,
    swing_sigma: float = 0.08,
    shoulde_joint_names: list[str] = ["left_shoulder_pitch_joint",
                    "right_shoulder_pitch_joint"]) -> torch.Tensor:

    #
    shoulde_ids = [env.dof_names.index(name) for name in shoulde_joint_names]
    #
    qpos = env.simulator.dof_pos[:, shoulde_ids]
    default_qpos = env.default_dof_pos[:, shoulde_ids]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")

    command_tensor = getattr(env.manager, "commands", None) if hasattr(env, "manager") else None

    if command_tensor is not None:
        swing_sign = torch.sign(command_tensor[:, :1])
        ##
        swing_target = torch.cos(gait_state.phase + torch.pi) * swing_range * swing_sign + cent_pos

        stand_mask = torch.logical_and(
            torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
            torch.abs(command_tensor[:, 2]) < 0.01,
        )
        if stand_mask.any():
            swing_target[stand_mask] = default_qpos[stand_mask]
    else:
        swing_target = default_qpos

    error = torch.square(qpos - swing_target)
    error = torch.sum(error, dim=-1)
    return torch.exp(-error / swing_sigma)


def penalty_knee(env, joint_names: list[str] = [
        "left_knee_joint",
        "right_knee_joint"]) -> torch.Tensor:

    #
    joint_ids = [env.dof_names.index(name) for name in joint_names]
    #
    qpos_error = (env.simulator.dof_pos[:, joint_ids]).clone()
    default_qpos = env.default_dof_pos[:, joint_ids]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    command_tensor = getattr(env.manager, "commands", None) if hasattr(env, "manager") else None

    if command_tensor is not None:
        ##
        phase = (gait_state.phase + torch.pi) / (2 * torch.pi)
        swing_phase = phase > 0.5

        qpos_error[swing_phase] = 0

        stand_mask = torch.logical_and(
            torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
            torch.abs(command_tensor[:, 2]) < 0.01,
        )
        if stand_mask.any():
            qpos_error[stand_mask] -= default_qpos[stand_mask]
    else:
        qpos_error -= default_qpos

    penalty_error = torch.sum(torch.square(qpos_error), dim=-1)
    return penalty_error
