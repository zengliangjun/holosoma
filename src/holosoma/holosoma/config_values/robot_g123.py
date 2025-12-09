from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotBridgeConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState,
)

g1_23dof = RobotConfig(
    num_bodies=26,
    dof_obs_size=23,
    actions_dim=23,
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_foot_contact_point", "right_foot_contact_point"],
    num_feet=2,
    foot_body_name="ankle_roll_link",
    foot_height_name="foot_contact_point",
    knee_name="knee_link",
    torso_name="waist_yaw_link",
    dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        #"waist_roll_joint",  13
        #"waist_pitch_joint", 14
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        #"left_wrist_pitch_joint", 20
        #"left_wrist_yaw_joint", 21
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        #"right_wrist_pitch_joint", 27
        #"right_wrist_yaw_joint", 28
    ],
    upper_dof_names=[
        "waist_yaw_joint",
        # "waist_roll_joint",
        # "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        # "left_wrist_pitch_joint",
        # "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        # "right_wrist_pitch_joint",
        # "right_wrist_yaw_joint",
    ],
    upper_left_arm_dof_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        #"left_wrist_pitch_joint",
        #"left_wrist_yaw_joint",
    ],
    upper_right_arm_dof_names=[
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        #"right_wrist_pitch_joint",
        #"right_wrist_yaw_joint",
    ],
    lower_dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    has_torso=True,
    has_upper_body_dof=True,
    left_ankle_dof_names=["left_ankle_pitch_joint", "left_ankle_roll_joint"],
    right_ankle_dof_names=["right_ankle_pitch_joint", "right_ankle_roll_joint"],
    knee_dof_names=["left_knee_joint", "right_knee_joint"],
    hips_dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
    ],
    dof_pos_lower_limit_list=[
        -2.5307,
        -0.5236,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.618,
        #-0.52,  13
        #-0.52,  14
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.972222054,
        # -1.61443, 20
        # -1.61443, 21
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.972222054,
        #-1.61443,  27
        #-1.61443,  28
    ],
    dof_pos_upper_limit_list=[
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        #0.52,  13
        #0.52,  14
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.972222054,
        #1.61443, 20
        #1.61443, 21
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.972222054,
        #1.61443,  27
        #1.61443,  28
    ],
    dof_vel_limit_list=[
        32.0,
        20.0,
        32.0,
        20.0,
        37.0,
        37.0,  # left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        32.0,
        20.0,
        32.0,
        20.0,
        37.0,
        37.0,  # right leg
        32.0,
        # 37.0,  13
        # 37.0,  14  # waist (yaw, roll, pitch)
        37.0,
        37.0,
        37.0,
        37.0,  # left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
        37.0,
        # 22.0, 20
        # 22.0, 21  # left wrist (roll, pitch, yaw)
        37.0,
        37.0,
        37.0,
        37.0,  # right arm
        37.0,
        # 22.0,  27
        # 22.0,  28
    ],  # right wrist
    dof_effort_limit_list=[
        88.0,
        139.0,
        88.0,
        139.0,
        50.0,
        50.0,  # left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        88.0,
        139.0,
        88.0,
        139.0,
        50.0,
        50.0,  # right leg
        88.0,
        #50.0,  13
        #50.0,  14  # waist (yaw, roll, pitch)
        25.0,
        25.0,
        25.0,
        25.0,  # left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
        25.0,
        # 5.0, 20
        # 5.0, 21  # left wrist (roll, pitch, yaw)
        25.0,
        25.0,
        25.0,
        25.0,  # right arm
        25.0,
        # 5.0,  27
        # 5.0,  28
    ],  # right wrist
    dof_armature_list=[
        0.010177520,
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
        0.003609725,
        0.003609725,
        0.003609725,
        0.003609725,  # left arm
        0.003609725,
        # 0.00425, 20
        # 0.00425, 21  # left wrist
        0.003609725,
        0.003609725,
        0.003609725,
        0.003609725,  # right arm
        0.003609725,
        # 0.00425,  27
        # 0.00425,  28
    ],  # right wrist
    dof_joint_friction_list=[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
    ],
    body_names=[
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "left_foot_contact_point",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_foot_contact_point",
        "waist_yaw_link",
        #"waist_roll_link",
        #"torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        #"left_wrist_pitch_link",
        #"left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        #"right_wrist_pitch_link",
        #"right_wrist_yaw_link",
    ],
    terminate_after_contacts_on=["pelvis", "shoulder", "hip"],
    penalize_contacts_on=["pelvis", "shoulder", "hip"],
    init_state=RobotInitState(
        pos=[0.0, 0.0, 0.8],  # x,y,z [m]
        rot=[0.0, 0.0, 0.0, 1.0],  # x,y,z,w [quat]
        lin_vel=[0.0, 0.0, 0.0],  # x,y,z [m/s]
        ang_vel=[0.0, 0.0, 0.0],  # x,y,z [rad/s]
        default_joint_angles={
            "left_hip_pitch_joint": -0.312,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.312,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.669,
            "right_ankle_pitch_joint": -0.363,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            # "waist_roll_joint": 0.0,
            # "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.0,
            # "left_wrist_pitch_joint": 0.0,
            # "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.6,
            "right_wrist_roll_joint": 0.0,
            # "right_wrist_pitch_joint": 0.0,
            # "right_wrist_yaw_joint": 0.0,
        },
    ),
    randomize_link_body_names=[
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
    ],
    waist_dof_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    waist_yaw_dof_name="waist_yaw_joint",
    waist_roll_dof_name="waist_roll_joint",
    waist_pitch_dof_name="waist_pitch_joint",
    arm_dof_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        # "left_wrist_pitch_joint",
        # "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        # "right_wrist_pitch_joint",
        # "right_wrist_yaw_joint",
    ],
    left_arm_dof_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        # "left_wrist_pitch_joint",
        # "left_wrist_yaw_joint",
    ],
    right_arm_dof_names=[
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        # "right_wrist_pitch_joint",
        # "right_wrist_yaw_joint",
    ],
    symmetry_joint_names={
        # Lower body joints
        "left_hip_pitch_joint": "right_hip_pitch_joint",
        "left_hip_roll_joint": "right_hip_roll_joint",
        "left_hip_yaw_joint": "right_hip_yaw_joint",
        "left_knee_joint": "right_knee_joint",
        "left_ankle_pitch_joint": "right_ankle_pitch_joint",
        "left_ankle_roll_joint": "right_ankle_roll_joint",
        "right_hip_pitch_joint": "left_hip_pitch_joint",
        "right_hip_roll_joint": "left_hip_roll_joint",
        "right_hip_yaw_joint": "left_hip_yaw_joint",
        "right_knee_joint": "left_knee_joint",
        "right_ankle_pitch_joint": "left_ankle_pitch_joint",
        "right_ankle_roll_joint": "left_ankle_roll_joint",
        # Upper body joints
        "left_shoulder_pitch_joint": "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint": "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint": "right_shoulder_yaw_joint",
        "left_elbow_joint": "right_elbow_joint",
        "left_wrist_roll_joint": "right_wrist_roll_joint",
        # "left_wrist_pitch_joint": "right_wrist_pitch_joint",
        "left_wrist_yaw_joint": "right_wrist_yaw_joint",
        "right_shoulder_pitch_joint": "left_shoulder_pitch_joint",
        "right_shoulder_roll_joint": "left_shoulder_roll_joint",
        "right_shoulder_yaw_joint": "left_shoulder_yaw_joint",
        "right_elbow_joint": "left_elbow_joint",
        "right_wrist_roll_joint": "left_wrist_roll_joint",
        # "right_wrist_pitch_joint": "left_wrist_pitch_joint",
        "right_wrist_yaw_joint": "left_wrist_yaw_joint",
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
        # Shoulder roll and yaw joints
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        # Wrist roll and yaw joints
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ],
    apply_dof_armature_in_isaacgym=True,
    contact_pairs_multiplier=16,
    control=RobotControlConfig(
        control_type="P",
        stiffness={
            "hip_yaw": 40.179238471,  # STIFFNESS_7520_14
            "hip_roll": 99.098427777,  # STIFFNESS_7520_22
            "hip_pitch": 40.179238471,  # STIFFNESS_7520_14
            "knee": 99.098427777,  # STIFFNESS_7520_22
            "ankle_pitch": 28.501246196,  # 2*STIFFNESS_5020
            "ankle_roll": 28.501246196,  # 2*STIFFNESS_5020
            "waist_yaw": 40.179238471,  # STIFFNESS_7520_14
            # "waist_roll": 28.501246196,  # 2*STIFFNESS_5020
            # "waist_pitch": 28.501246196,  # 2*STIFFNESS_5020
            "shoulder_pitch": 14.250623098,  # STIFFNESS_5020
            "shoulder_roll": 14.250623098,  # STIFFNESS_5020
            "shoulder_yaw": 14.250623098,  # STIFFNESS_5020
            "elbow": 14.250623098,  # STIFFNESS_5020
            "wrist_roll": 14.250623098,  # STIFFNESS_5020
            # "wrist_pitch": 16.778327481,  # STIFFNESS_4010
            # "wrist_yaw": 16.778327481,  # STIFFNESS_4010
        },
        damping={
            "hip_yaw": 2.557889765,  # DAMPING_7520_14
            "hip_roll": 6.308801854,  # DAMPING_7520_22
            "hip_pitch": 2.557889765,  # DAMPING_7520_14
            "knee": 6.308801854,  # DAMPING_7520_22
            "ankle_pitch": 1.814445687,  # 2*DAMPING_5020
            "ankle_roll": 1.814445687,  # 2*DAMPING_5020
            "waist_yaw": 2.557889765,  # DAMPING_7520_14
            # "waist_roll": 1.814445687,  # 2*DAMPING_5020
            # "waist_pitch": 1.814445687,  # 2*DAMPING_5020
            "shoulder_pitch": 0.907222843,  # DAMPING_5020
            "shoulder_roll": 0.907222843,  # DAMPING_5020
            "shoulder_yaw": 0.907222843,  # DAMPING_5020
            "elbow": 0.907222843,  # DAMPING_5020
            "wrist_roll": 0.907222843,  # DAMPING_5020
            # "wrist_pitch": 1.068141502,  # DAMPING_4010
            # "wrist_yaw": 1.068141502,  # DAMPING_4010
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
        urdf_file="g1/g1_23dof.urdf",
        usd_file=None,
        xml_file="g1/g1_23dof.xml",
        robot_type="g1_23dof",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=False,
    ),
    bridge=RobotBridgeConfig(
        sdk_type="unitree",
        motor_type="serial",
    ),
)
