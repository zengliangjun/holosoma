"""Locomotion curriculum presets for the G1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.00025,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

g1_curriculum_fast_sac = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.3,
                "min_scale": 0.3,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 600.0,
                "degree": 0.0001,
            },
        ),
        "domain_rand_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion_ext:DomainRandCurriculum",
            params={
                "enabled": True,
                "initial_scale": 0.05,
                "min_scale": 0.05,
                "max_scale": 1.0,
                "level_down_threshold": 750.0,
                "level_up_threshold": 850.0,
                "degree": 0.0001,
                "dr_reset_names": ["push_randomizer_state"],
                #"mass_randomizer",
                #"randomize_friction",
                #"randomize_base_com"
                #                 ],

                "dr_push_randomizer_state_params": ["max_push_vel"],
                "dr_push_randomizer_state_max_push_vel_init": [0.3, 0.3],
                "dr_push_randomizer_state_max_push_vel_final": [1.5, 1.5],  # [4.5, 4.5],

                "dr_mass_randomizer_params": ["link_mass_range", "added_mass_range"],
                "dr_mass_randomizer_link_mass_range_init": [0.95, 1.05],
                "dr_mass_randomizer_link_mass_range_final": [0.6, 1.5],
                "dr_mass_randomizer_added_mass_range_init": [0, 0.5],
                "dr_mass_randomizer_added_mass_range_final": [0, 6.0],

                "dr_randomize_friction_params": ["friction_range"],
                "dr_randomize_friction_friction_range_init": [0.95, 1.05],
                "dr_randomize_friction_friction_range_final": [0.4, 1.5],

                "dr_randomize_base_com_params": ["base_com_range.x", "base_com_range.y", "base_com_range.z"],
                "dr_randomize_base_com_base_com_range_x_init": [-0.005, 0.005],
                "dr_randomize_base_com_base_com_range_x_final": [-0.05, 0.05],
                "dr_randomize_base_com_base_com_range_y_init": [-0.005, 0.005],
                "dr_randomize_base_com_base_com_range_y_final": [-0.05, 0.05],
                "dr_randomize_base_com_base_com_range_z_init": [-0.005, 0.005],
                "dr_randomize_base_com_base_com_range_z_final": [-0.05, 0.05],
            },
        )
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["g1_curriculum", "g1_curriculum_fast_sac"]
