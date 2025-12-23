"""Locomotion reward presets for the G1 robot."""
from __future__ import annotations
import dataclasses

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_23dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "penalty_pose_maxoffset": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_pose_maxoffset",
            weight=-5.0,
            params={
                "joint_names": [
                    "left_hip_pitch_joint",
                    "left_knee_joint",
                    "right_hip_pitch_joint",
                    "right_knee_joint"
                ],
                "max_offset": [
                    0.3,
                    0.25,
                    0.3,
                    0.25,
                ]
            },
            tags=["penalty_curriculum"],
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01,  # hip_pitch
                    1.0,   # hip_roll
                    5.0,   # hip_yaw
                    0.01,  #    knee
                    5.0,   #    ankle_pitch
                    5.0,   #    ankle_roll
                    0.01,  # hip_pitch
                    1.0,   # hip_roll
                    5.0,   # hip_yaw
                    0.01,  #    knee
                    5.0,
                    5.0,
                    50.0,
                    # 50.0,  13
                    # 50.0,  14
                    50.0,     #    shoulder_pitch
                    50.0,    #    shoulder_roll
                    50.0,    #    shoulder_yaw
                    50.0,     #  elbow
                    50.0,    #  wrist_roll
                    # 50.0, 20
                    # 50.0, 21
                    50.0,     #    shoulder_pitch
                    50.0,    #    shoulder_roll
                    50.0,    #    shoulder_yaw
                    50.0,     #  elbow
                    50.0,    #  wrist_roll
                    # 50.0,  27
                    # 50.0,  28
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_contact_forces": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_feet_contact_forces_v1",
            weight=-0.005,
            params={"force_threshold": 500,
                    "max_force": 400},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

__all__ = ["g1_23dof_loco_fast_sac"]
