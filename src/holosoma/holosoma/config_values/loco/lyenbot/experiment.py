from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    robot_lyenbot,
    robot_lyenbotlegs,
    simulator,
    termination,
    terrain,
)

from holosoma.config_values.loco.lyenbot import (
    command,
    observation,
    randomization,
    curriculum,
    reward
)

lyenbotloc_fastsac_v1 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="lyenbot", name="lyenbot_loc_fastsac_v1"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config,
            num_learning_iterations=200000, use_symmetry=True)),

    simulator=simulator.isaacgym,
    robot=robot_lyenbot.lyenbot_23dof,
    terrain=terrain.terrain_locomotion_mix_dof23,
    observation=observation.lyenbot_loco,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.lyenbot_randomization,
    command=command.lyenbot_command,
    curriculum=curriculum.lyenbot_curriculum_fast_sac,
    reward=reward.lyenbot_loco_fast_sac_v1,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)


lyenbotlegloc_fastsac_v1 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="lyenbot", name="lyenbotlegloc_loc_fastsac_v1"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config,
            num_learning_iterations=100000, use_symmetry=True)),

    simulator=simulator.isaacgym,
    robot=robot_lyenbotlegs.lyenbotleg,
    terrain=terrain.terrain_locomotion_mix_dof23,
    observation=observation.lyenbot_loco,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.lyenbot_randomization,
    command=command.lyenbot_command,
    curriculum=curriculum.lyenbot_curriculum_fast_sac,
    reward=reward.lyenbotlegs_loco_fast_sac_v1,
    nightly=NightlyConfig(
        iterations=100000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)
