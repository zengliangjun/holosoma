from __future__ import annotations

from .locomotion import LocomotionCommand, LocomotionGait
from holosoma.utils.torch_utils import torch_rand_float

from typing import Any, cast
import torch


class CommandEx(LocomotionCommand):
    """Stateful command term that owns command buffers and resampling logic."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        if hasattr(self, "stand_prob"):
            delattr(self, "stand_prob")

        stand_prob: float = float(params.get("stand_prob", 0.1))
        walk_prob: float = float(params.get("walk_prob", 0.23))
        spin_prob: float = float(params.get("spin_prob", 0.23))
        walk_spin_prob: float = float(params.get("walk_spin_prob", 0.44))

        assert 1 == stand_prob + walk_prob + spin_prob + walk_spin_prob
        self.probs = torch.tensor([stand_prob, walk_prob, spin_prob, walk_spin_prob])

        self.stand_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.walk_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.spin_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.walk_spin_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    def _resample(self, env_ids: torch.Tensor) -> None:
        self.stand_mask[env_ids] = False
        self.walk_mask[env_ids] = False
        self.spin_mask[env_ids] = False
        self.walk_spin_mask[env_ids] = False

        ##
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return

        device = self.env.device
        ranges = self.command_ranges

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0],
            ranges["lin_vel_x"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0],
            ranges["lin_vel_y"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            ranges["ang_vel_yaw"][0],
            ranges["ang_vel_yaw"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            gait_state = manager.get_state("locomotion_gait")
        else:
            gait_state = None

        if gait_state is not None:
            cast("LocomotionGait", gait_state).resample_frequency(env_ids)

        indices = torch.multinomial(self.probs, num_samples=env_ids.shape[0], replacement=True)
        for cls in range(self.probs.shape[0]):
            # 筛选出当前类别的所有样本索引，再提取对应数据
            cls_mask = (indices == cls)
            if 0 == cls:  # stand
                stand_ids = env_ids[cls_mask]  #
                if 0 == len(stand_ids):
                    continue
                commands[stand_ids, :3] = 0.0
                self.stand_mask[stand_ids] = True

            elif 1 == cls:  # walk
                walk_ids = env_ids[cls_mask]  #
                if 0 == len(walk_ids):
                    continue
                commands[walk_ids, 2:] = 0.0
                self.walk_mask[walk_ids] = True

            elif 2 == cls:  # spin
                spin_ids = env_ids[cls_mask]  #
                if 0 == len(spin_ids):
                    continue
                commands[spin_ids, :2] = 0.0
                self.spin_mask[spin_ids] = True

            elif 3 == cls:  # walk spin
                walk_spin_ids = env_ids[cls_mask]  #
                if 0 == len(walk_spin_ids):
                    continue
                self.walk_spin_mask[walk_spin_ids] = True

        stand_mas = torch.norm(commands[env_ids], dim=-1) < 0.1
        stand_ids = env_ids[stand_mas]  #
        if 0 != len(stand_ids):
            commands[stand_ids] = 0
            self.stand_mask[stand_ids] = True
            self.walk_mask[stand_ids] = False
            self.spin_mask[stand_ids] = False
            self.walk_spin_mask[stand_ids] = False
