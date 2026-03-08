"""Locomotion reward presets for the G1 robot."""
from __future__ import annotations
import dataclasses

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


lyenbot_loco_fast_sac_v1 = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=3.1,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.14,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_linear_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_linear_vel_z",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # "penalty_ang_vel_xy": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
        #     weight=-1.0,
        #     params={},
        #     tags=["penalty_curriculum"],
        # ),
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
        # "penalty_joint_vel": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion_ext:penalty_joint_vel",
        #     weight=-0.05,
        #     params={},
        #     tags=["penalty_curriculum"],
        # ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=6.8,
            params={"swing_height": 0.12, "tracking_sigma": 0.008},
        ),
        # "feet_gait": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion_ext:feet_gait",
        #     weight=1.0,
        #     params={"threshold": 0.5},
        # ),
        "penalty_pose_maxoffset": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_pose_maxoffset",
            weight=-0.5,
            params={
                "joint_names": [
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    'left_shoulder_pitch_joint',
                    'right_shoulder_pitch_joint',
                ],
                "max_offset": [
                    0.25,
                    0.25,
                    0.25,
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
                    0.01,
                    1.0,   # 25.0,  # hip_roll
                    5.0,   # 50.0,  # 5.0,  hip_yaw
                    0.01,
                    5.0,   # ankle_pitch
                    5.0,   # 50.0,  # 5.0, ankle_roll
                    0.01,
                    1.0,   # 25.0,  # 1.0,  hip_roll
                    5.0,   # 50.0,  # 5.0,  hip_yaw
                    0.01,
                    5.0,   # ankle_pitch
                    5.0,   # 50.0,  # 5.0, ankle_roll
                    50.0,
                    # 50.0,  13
                    # 50.0,  14
                    35.0,  # 50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    # 50.0, 20
                    # 50.0, 21
                    35.0,  # 50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    # 50.0,  27
                    # 50.0,  28
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_contact_forces": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_feet_contact_forces_v1",
            weight=-1e-3,
            params={"force_threshold": 500,
                    "max_force": 800},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.13},
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


lyenbotlegs_loco_fast_sac_v1 = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=3.1,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.14,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_linear_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_linear_vel_z",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # "penalty_ang_vel_xy": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
        #     weight=-1.0,
        #     params={},
        #     tags=["penalty_curriculum"],
        # ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-15.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # "penalty_ankle_rate": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion_ext:penalty_action_rate2",
        #     weight=-2.0,
        #     params={
        #         "joint_names": [
        #             "left_ankle_pitch_joint",
        #             "right_ankle_pitch_joint",
        #             "left_ankle_roll_joint",
        #             "right_ankle_roll_joint",
        #         ]},
        #     tags=["penalty_curriculum"],
        # ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # "penalty_joint_vel": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion_ext:penalty_joint_vel",
        #     weight=-0.05,
        #     params={},
        #     tags=["penalty_curriculum"],
        # ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=6.8,
            params={"swing_height": 0.045, "tracking_sigma": 0.008},
        ),
        # "feet_gait": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion_ext:feet_gait",
        #     weight=1.0,
        #     params={"threshold": 0.5},
        # ),
        "penalty_pose_maxoffset": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_pose_maxoffset",
            weight=-5,
            params={
                "joint_names": [
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                ],
                "max_offset": [
                    0.25,
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
                    0.01,
                    5.0,   # 25.0,  # hip_roll
                    10.0,   # 50.0,  # 5.0,  hip_yaw
                    0.01,
                    5.0,   # ankle_pitch
                    10.0,   # 50.0,  # 5.0, ankle_roll
                    0.01,
                    5.0,   # 25.0,  # 1.0,  hip_roll
                    10.0,   # 50.0,  # 5.0,  hip_yaw
                    0.01,
                    5.0,   # ankle_pitch
                    10.0,   # 50.0,  # 5.0, ankle_roll
                    50.0,
                    # 50.0,  13
                    # 50.0,  14
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_contact_forces": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_ext:penalty_feet_contact_forces_v1",
            weight=-1e-3,
            params={"force_threshold": 500,
                    "max_force": 800},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.18},
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

__all__ = ["lyenbot_loco_fast_sac_v1", "lyenbotlegs_loco_fast_sac_v1"]
