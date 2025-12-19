
from __future__ import annotations
import dataclasses

from holosoma_inference.config.config_types.robot import RobotConfig

g1_23dof = RobotConfig(
    # Identity
    robot_type="g1_23dof",
    robot="g1",

    # SDK Configuration
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    use_sensor=False,

    # Dimensions
    num_motors=23,
    num_joints=23,
    num_upper_body_joints=10,

    # Default Positions
    default_dof_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, #0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, #0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, #0.0, 0.0,  # right arm
    ),
    default_motor_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, #0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, #0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, #0.0, 0.0,  # right arm
    ),

    # Limits
    joint_pos_min=(
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,  # left leg
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,  # right leg
        -2.618, #-0.52, -0.52,  # waist
        -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, #-1.61443, -1.61443,  # left arm
        -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, #-1.61443, -1.61443,  # right arm
    ),
    joint_pos_max=(
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,  # left leg
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,  # right leg
        2.618, #0.52, 0.52,  # waist
        2.6704, 2.2515, 2.618, 2.0944, 1.972222054, #1.61443, 1.61443,  # left arm
        2.6704, 1.5882, 2.618, 2.0944, 1.972222054, #1.61443, 1.61443,  # right arm
    ),
    joint_vel_limit=(
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # left leg
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # right leg
        32.0, # 37.0, 37.0,  # waist
        37.0, 37.0, 37.0, 37.0, 37.0, # 22.0, 22.0,  # left arm
        37.0, 37.0, 37.0, 37.0, 37.0, # 22.0, 22.0,  # right arm
    ),
    motor_effort_limit=(
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # left leg
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # right leg
        88.0, # 50.0, 50.0,  # waist
        25.0, 25.0, 25.0, 25.0, 25.0, # 5.0, 5.0,  # left arm
        25.0, 25.0, 25.0, 25.0, 25.0, # 5.0, 5.0,  # right arm
    ),

    # Mappings
    motor2joint=tuple(range(23)),  # Identity mapping
    joint2motor=tuple(range(23)),  # Identity mapping
    dof_names=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", #"waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", #"left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", #"right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_upper_body=(
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", #"left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", #"right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", #"waist_roll_joint", "waist_pitch_joint",
    ),

    # Link Names
    torso_link_name="waist_yaw_link",
    left_hand_link_name="left_rubber_hand",
    right_hand_link_name="right_rubber_hand",

    # Unitree-Specific Constants
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    weak_motor_joint_index={
        "left_hip_yaw_joint": 0, "left_hip_roll_joint": 1, "left_hip_pitch_joint": 2,
        "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
        "right_hip_yaw_joint": 6, "right_hip_roll_joint": 7, "right_hip_pitch_joint": 8,
        "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
        "waist_yaw_joint": 12, # "waist_roll_joint": 13, "waist_pitch_joint": 14,
        "left_shoulder_pitch_joint": 15 - 2, "left_shoulder_roll_joint": 16 - 2,
        "left_shoulder_yaw_joint": 17 - 2, "left_elbow_joint": 18 - 2,
        "left_wrist_roll_joint": 19 - 2, #"left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
        "right_shoulder_pitch_joint": 22 - 4, "right_shoulder_roll_joint": 23 - 4,
        "right_shoulder_yaw_joint": 24 - 4, "right_elbow_joint": 25 - 4,
        "right_wrist_roll_joint": 26 - 4, #"right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
    },
    #weak_motor_joint_index={
    #    "left_hip_yaw_joint": 0, "left_hip_roll_joint": 1, "left_hip_pitch_joint": 2,
    #    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    #    "right_hip_yaw_joint": 6, "right_hip_roll_joint": 7, "right_hip_pitch_joint": 8,
    #    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    #    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    #    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
    #    "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
    #    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
    #    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    #    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
    #    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
    #},
    motion={"body_name_ref": ["torso_link"]},
)


g1_23dof_dict = dataclasses.asdict(g1_23dof)

g1_23dof_dict.update({
    "robot_type": "rllab_init_g1_23dof",
    "default_dof_angles": [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg
        0.0, #0.0, 0.0,  # waist
        0.3,  0.25, 0.0, 0.97, 0.15, #0.0, 0.0,  # left arm
        0.3, -0.25, 0.0, 0.97, 0.15, #0.0, 0.0,  # right arm
    ]
    })

unitree_rl_init_g1_23dof = RobotConfig(**g1_23dof_dict)
