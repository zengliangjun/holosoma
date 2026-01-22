import numpy as np
import os
import yaml
import torch
import dataclasses

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types import env as envpkg
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.observation import ObsGroupCfg


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(algo: BaseAlgo, task: BaseTask, env_config: envpkg.EnvConfig, log_dir: str):
    joint_sdk_names = env_config.robot.dof_names
    joint_ids_map = np.arange(0, len(joint_sdk_names), 1, dtype=int)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map.tolist()
    cfg["step_dt"] = task.dt
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = task.p_gains.detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = task.d_gains.detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = task.default_dof_pos_base[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if "locomotion_command" in env_config.command.setup_terms:  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}

        termCfg: CommandTermCfg = env_config.command.setup_terms["locomotion_command"]

        if "command_ranges" in termCfg.params:
            ranges = termCfg.params["command_ranges"]
            if "heading" in ranges:
                ranges.pop("heading")
            for item_name in ["lin_vel_x", "lin_vel_y"]:
                ranges[item_name] = list(ranges[item_name])

            ranges["ang_vel_z"] = list(ranges.pop("ang_vel_yaw"))
            cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = task.action_manager.active_terms
    action_terms = zip(action_names, task.action_manager._term_instances.values())
    action_scale: float = env_config.robot.control.action_scale

    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg: dict = dataclasses.asdict(action_term.cfg)

        if isinstance(term_cfg["scale"], float):
            scale = term_cfg["scale"] * action_scale
            term_cfg["scale"] = [scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg["scale"] = action_term.action_scales.detach().cpu().numpy().tolist()

        #if env_config.robot.control.action_clip_value:
        #    term_cfg["clip"] = [env_config.robot.control.action_clip_value for _ in range(action_term.action_dim)]
        term_cfg["clip"] = None

        term_cfg["offset"] = task.default_dof_pos_base[0].detach().cpu().numpy().tolist()

        for _ in ["class_type", "func", "params"]:
            if _ in term_cfg:
                term_cfg.pop(_)

        cfg["actions"][action_name] = term_cfg
        cfg["actions"][action_name]["joint_ids"] = joint_ids_map.tolist()
        cfg["actions"][action_name]["joint_names"] = joint_sdk_names

    # --- observations ---
    obs_cfgs: ObsGroupCfg = task.observation_manager.cfg.groups["actor_obs"]
    history_length = obs_cfgs.history_length
    cfg["observations"] = {}

    for term_name, term_cfg in obs_cfgs.terms.items():
        obs_tensor = task.observation_manager._compute_term("actor_obs", term_name, term_cfg)

        obs_dims = tuple(obs_tensor.shape)
        term_dict = {}
        if term_cfg.scale is not None:
            scale = term_cfg.scale
            if isinstance(scale, float):
                term_dict["scale"] = [scale for _ in range(obs_dims[1])]
            else:
                term_dict["scale"] = scale
        else:
            term_dict["scale"] = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if history_length == 0:
            term_dict["history_length"] = 1
        else:
            term_dict["history_length"] = history_length

        term_dict["params"] = {}
        term_dict["clip"] = None

        cfg["observations"][term_name] = term_dict

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
