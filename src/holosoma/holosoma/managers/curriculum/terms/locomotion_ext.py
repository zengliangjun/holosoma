"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from holosoma.managers.curriculum.base import CurriculumTermBase

class DomainRandCurriculum(CurriculumTermBase):
    """Stateful penalty curriculum that scales reward term weights based on episode length.

    This curriculum term adaptively scales penalty reward weights during training.
    When episodes are short (robot falls quickly), penalties are reduced to make
    learning easier. As episodes get longer (robot stays up), penalties gradually
    increase to refine behavior.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)

        # Get parameters from config
        params = cfg.params
        self.enabled = params.get("enabled", True)
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))

        self.level_down_threshold = float(params.get("level_down_threshold", 750.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 850.0))
        self.degree = float(params.get("degree", 0.0))

        # State variables (previously stored on env)
        self.current_scale = float(params.get("initial_scale", 1.0))
        self.dr_reset_names = params.get("dr_reset_names", [])
        self.dr_step_names = params.get("dr_step_names", [])


    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled or not hasattr(self.env, "randomization_manager"):
            return

        # self._update_settings()

        # Set flag for logging compatibility
        self.env.use_domain_rand_scale_curriculum = True
        #self.env.domain_rand_scale = self.current_scale

    def reset(self, env_ids) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "randomization_manager"):
            return

        if len(self.dr_reset_names) == 0 and \
            len(self.dr_step_names) == 0:
            return

        average_length = float(self.env.average_episode_length)

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        self._update_states(self.dr_reset_names)
        self._update_settings(self.dr_reset_names, self.env.randomization_manager._reset_names, self.env.randomization_manager._reset_cfgs)

        # Update for logging
        #self.env.domain_rand_scale = self.current_scale

        # Update log_dict for WandB logging
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["domain_rand_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if len(self.dr_reset_names) == 0 and \
            len(self.dr_step_names) == 0:
            return


        self._update_states(self.dr_step_names)

        self._update_settings(self.dr_step_names, self.env.randomization_manager._step_names, self.env.randomization_manager._step_cfgs)


    def _update_settings(self, curriculum_names, dr_names, dr_cfgs):
        params = self.cfg.params
        # Identify penalty rewards by tag using public reward manager APIs

        for curriculum_name in curriculum_names:
            id = dr_names.index(curriculum_name)
            if -1 == id:
                continue

            term_cfg = dr_cfgs[id]

            term_cfg_dict = dataclasses.asdict(term_cfg)

            _update_dict(term_cfg_dict, params, curriculum_name, self.current_scale)

            update_term_cfg = term_cfg.__class__(**term_cfg_dict)
            dr_cfgs[id] = update_term_cfg

    def _update_states(self, curriculum_names):
        params = self.cfg.params
        # Identify penalty rewards by tag using public reward manager APIs
        from holosoma.managers.randomization.base import RandomizationTermBase

        for curriculum_name in curriculum_names:

            state: RandomizationTermBase = self.env.randomization_manager.get_state(curriculum_name)
            if state is None:
                continue

            term_param_names = params.get(f'dr_{curriculum_name}_params', [])

            term_update_params = {}

            for param_name in term_param_names:
                init_settings = params.get(f'dr_{curriculum_name}_{param_name}_init')
                final_settings = params.get(f'dr_{curriculum_name}_{param_name}_final')

                if isinstance(init_settings, list):
                    calcute_settings = []
                    for init, final in zip(init_settings, final_settings):
                        calcute_settings.append(init + (final - init) * self.current_scale)
                else:
                    calcute_settings = init_settings + (final_settings - init_settings) * self.current_scale

                term_update_params[param_name] = calcute_settings
            state.configure(**term_update_params)


def _update_dict(term_cfg_dict, params, curriculum_name, current_scale):

    curriculum_param_names = params.get(f'dr_{curriculum_name}_params', [])
    for param_name in curriculum_param_names:
        if -1 != param_name.find('.'):
            sub_key = True
            key_name = param_name.replace(".", "_")
        else:
            sub_key = False
            key_name = param_name

        init_settings = params.get(f'dr_{curriculum_name}_{key_name}_init')
        final_settings = params.get(f'dr_{curriculum_name}_{key_name}_final')

        if isinstance(init_settings, list):
            calcute_settings = []
            for init, final in zip(init_settings, final_settings):
                calcute_settings.append(init + (final - init) * current_scale)
        else:
            calcute_settings = init_settings + (final_settings - init_settings) * current_scale

        if sub_key:
            sub_names = param_name.split('.')
            sub_dict = term_cfg_dict['params']
            for sub_id, subname in enumerate(sub_names):
                if sub_id != len(sub_names) - 1:
                    sub_dict = sub_dict[subname]
                else:
                    sub_dict[subname] = calcute_settings
        else:
            term_cfg_dict['params'][param_name] = calcute_settings
