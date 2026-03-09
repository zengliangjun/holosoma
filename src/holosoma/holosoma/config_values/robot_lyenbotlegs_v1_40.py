from __future__ import annotations

from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotBridgeConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState
)

lyenbotleg = RobotConfig(
    num_bodies=16,
    dof_obs_size=13,
    actions_dim=13,
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_foot_contact_point", "right_foot_contact_point"],
    num_feet=2,
    foot_body_name="ankle_roll_link",
    foot_height_name="foot_contact_point",
    knee_name="knee_pitch_link",
    torso_name="torso",
    dof_names=[
        'left_hip_pitch_joint',
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_knee_pitch_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_pitch_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_knee_pitch_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',

        'waist_yaw_joint',

    ],
    upper_dof_names=[
        "waist_yaw_joint",

    ],
    upper_left_arm_dof_names=[
    ],
    upper_right_arm_dof_names=[
    ],
    lower_dof_names=[
        'left_hip_pitch_joint',
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_knee_pitch_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_pitch_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_knee_pitch_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
    ],
    has_torso=True,
    has_upper_body_dof=True,
    left_ankle_dof_names=["left_ankle_pitch_joint", "left_ankle_roll_joint"],
    right_ankle_dof_names=["right_ankle_pitch_joint", "right_ankle_roll_joint"],
    knee_dof_names=["left_knee_pitch_joint", "right_knee_pitch_joint"],
    hips_dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
    ],

    dof_pos_lower_limit_list=[
        -2.617,  # hip_pitch_joint
        -0.314,  # hip_roll_joint
        -2.617,  # hip_yaw_joint
        0.08726699650287628,     # knee_pitch_joint
        -0.534,  # -0.611,  # ankle_pitch_joint
        -0.1326,  # -0.314,  # ankle_roll_joint

        -2.617,
        -2.268,
        -2.617,
        0.08726699650287628,
        -0.534,  # -0.611,
        -0.1326,  # -0.314,

        -2.617,  # waist_yaw_joint
    ],

    dof_pos_upper_limit_list=[
        2.617,  # hip_pitch_joint
        2.268,  # hip_roll_joint
        2.617,  # hip_yaw_joint
        2.477,  # knee_pitch_joint
        0.534,   #  0.611,  # ankle_pitch_joint
        0.1326,  # 0.314,  # ankle_roll_joint

        2.617,
        0.314,
        2.617,
        2.477,
        0.534,   #  0.611,
        0.1326,  # 0.314,

        2.617,  # waist_yaw_joint

    ],

    dof_vel_limit_list=[
        12.04,  # 'left_hip_pitch_joint',
        17.79,  # 'left_hip_roll_joint',
        17.79,  # 'left_hip_yaw_joint',
        12.04,  # 'left_knee_pitch_joint',
        16.22,  # 'left_ankle_pitch_joint',
        16.22,  # 'left_ankle_roll_joint',
        12.04,  # 'right_hip_pitch_joint',
        17.79,  # 'right_hip_roll_joint',
        17.79,  # 'right_hip_yaw_joint',
        12.04,  # 'right_knee_pitch_joint',
        16.22,  # 'right_ankle_pitch_joint',
        16.22,  # 'right_ankle_roll_joint',

        17.79,  # 'waist_yaw_joint',

    ],  # right wrist
    dof_effort_limit_list=[
        140,  # 'left_hip_pitch_joint',
        80,  # 'left_hip_roll_joint',
        80,  # 'left_hip_yaw_joint',
        140,  # 'left_knee_pitch_joint',
        35,  # 'left_ankle_pitch_joint',
        35,  # 'left_ankle_roll_joint',
        140,  # 'right_hip_pitch_joint',
        80,  # 'right_hip_roll_joint',
        80,  # 'right_hip_yaw_joint',
        140,  # 'right_knee_pitch_joint',
        35,  # 'right_ankle_pitch_joint',
        35,  # 'right_ankle_roll_joint',

        80,  # 'waist_yaw_joint',

    ],  # right wrist
    dof_armature_list=[
        0.010177520,  # 'left_hip_pitch_joint',
        0.025101925,
        0.010177520,
        0.025101925,
        0.007219450,
        0.007219450,  # left leg
        0.010177520,
        0.025101925,
        0.010177520,
        0.025101925,
        0.007219450,
        0.007219450,  # right leg
        0.010177520,
        # 0.007219450,  13
        # 0.007219450,  14  # waist

    ],  # right wrist
    dof_joint_friction_list=[
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    body_names=[
        "base_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_pitch_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "left_foot_contact_point",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_pitch_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_foot_contact_point",
        "torso",
    ],
    terminate_after_contacts_on=["base_link", "hip", "knee"],
    penalize_contacts_on=["base_link", "hip", "knee"],
    init_state=RobotInitState(
        pos=[0.0, 0.0, 0.96],  # x,y,z [m]
        rot=[0.0, 0.0, 0.0, 1.0],  # x,y,z,w [quat]
        lin_vel=[0.0, 0.0, 0.0],  # x,y,z [m/s]
        ang_vel=[0.0, 0.0, 0.0],  # x,y,z [rad/s]
        default_joint_angles={
            "left_hip_pitch_joint": -0.1,  # -0.312,
            "left_hip_roll_joint": 0.05,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": 0.3,  # 0.669,
            "left_ankle_pitch_joint": -0.2,  # -0.363,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,  # -0.312,
            "right_hip_roll_joint": 0.05,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": .3,  # 0.669,
            "right_ankle_pitch_joint": -0.2,  # -0.363,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
        },
    ),
    randomize_link_body_names=[
        "base_link",
        'left_hip_pitch_link',
        'left_hip_roll_link',
        'left_hip_yaw_link',
        'left_knee_pitch_link',
        'left_ankle_pitch_link',
        'left_ankle_roll_link',

        'right_hip_pitch_link',
        'right_hip_roll_link',
        'right_hip_yaw_link',
        'right_knee_pitch_link',
        'right_ankle_pitch_link',
        'right_ankle_roll_link',

        'torso',
    ],
    waist_dof_names=["waist_yaw_joint"],
    waist_yaw_dof_name="waist_yaw_joint",
    # waist_roll_dof_name="waist_roll_joint",
    # waist_pitch_dof_name="waist_pitch_joint",
    arm_dof_names=[
    ],
    left_arm_dof_names=[
    ],
    right_arm_dof_names=[
    ],
    symmetry_joint_names={
        # Lower body joints
        "left_hip_pitch_joint": "right_hip_pitch_joint",
        "left_hip_roll_joint": "right_hip_roll_joint",
        "left_hip_yaw_joint": "right_hip_yaw_joint",
        "left_knee_pitch_joint": "right_knee_pitch_joint",
        "left_ankle_pitch_joint": "right_ankle_pitch_joint",
        "left_ankle_roll_joint": "right_ankle_roll_joint",
        "right_hip_pitch_joint": "left_hip_pitch_joint",
        "right_hip_roll_joint": "left_hip_roll_joint",
        "right_hip_yaw_joint": "left_hip_yaw_joint",
        "right_knee_pitch_joint": "left_knee_pitch_joint",
        "right_ankle_pitch_joint": "left_ankle_pitch_joint",
        "right_ankle_roll_joint": "left_ankle_roll_joint",

        # Central joints (map to themselves)
        "waist_yaw_joint": "waist_yaw_joint",
        # "waist_roll_joint": "waist_roll_joint",
        # "waist_pitch_joint": "waist_pitch_joint",
    },
    flip_sign_joint_names=[
        # Hip roll and yaw joints
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        # Ankle roll joints
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        # Waist roll and yaw joints
        # "waist_roll_joint",
        "waist_yaw_joint",
    ],
    apply_dof_armature_in_isaacgym=True,
    contact_pairs_multiplier=16,
    control=RobotControlConfig(
        control_type="P",
        stiffness={
            "hip_yaw": 80,  # STIFFNESS_7520_14
            "hip_roll": 80,  # STIFFNESS_7520_22
            "hip_pitch": 80,  # STIFFNESS_7520_14
            "knee": 60,  # STIFFNESS_7520_22
            "ankle_pitch": 40,  # 2*STIFFNESS_5020
            "ankle_roll": 40,  # 2*STIFFNESS_5020
            "waist_yaw": 60,  # STIFFNESS_7520_14
        },
        damping={
            "hip_yaw": 4,  # DAMPING_7520_14
            "hip_roll": 4,  # DAMPING_7520_22
            "hip_pitch": 4,  # DAMPING_7520_14
            "knee": 3,  # DAMPING_7520_22
            "ankle_pitch": 2,  # 2*DAMPING_5020
            "ankle_roll": 2,  # 2*DAMPING_5020
            "waist_yaw": 3,  # DAMPING_7520_14
        },
        action_scale=0.25,  # 0.25 for locomotion, 1.0 for whole body tracking
        action_clip_value=100.0,
        clip_actions=True,
        clip_torques=True,
    ),
    asset=RobotAssetConfig(
        asset_root="@holosoma/data/robots",
        collapse_fixed_joints=True,
        replace_cylinder_with_capsule=True,
        flip_visual_attachments=False,
        armature=0.001,
        thickness=0.01,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        angular_damping=0.0,
        linear_damping=0.0,
        urdf_file="lyenbot/lyenbotlegs-A_E11-260308_collision35.urdf",
        usd_file=None,
        xml_file="lyenbot/lyenbotlegs-A_E11-260308_collision.xml",
        robot_type="lyenbot",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=False,
    ),
    bridge=RobotBridgeConfig(
        sdk_type="unitree",
        motor_type="serial",
    ),
)
