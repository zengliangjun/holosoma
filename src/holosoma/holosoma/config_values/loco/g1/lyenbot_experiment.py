from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    robot_lyenbot,
    simulator,
    termination,
    terrain,
)

from holosoma.config_values.loco.g1 import (
    dof23_curriculum,
    dof23_command,
    dof23_observation,
    dof23_randomization,
    dof23_reward,
)


lyenbot_loc_fastsac_v2 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="lyenbot_loc", name="lyenbot_fastsac_v2"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config,
            num_learning_iterations=100000, use_symmetry=True)),

    simulator=simulator.isaacgym,
    robot=robot_lyenbot.lyenbot_23dof,
    terrain=terrain.terrain_locomotion_mix_dof23,
    observation=dof23_observation.g1_loco,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=dof23_randomization.g1_randomization,
    command=dof23_command.g1_command,
    curriculum=dof23_curriculum.g1_curriculum_fast_sac,
    reward=dof23_reward.g1_23dof_shoulder_gait,
    nightly=NightlyConfig(
        iterations=100000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)
