import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, \
        g1_29dof_fast_sac, \
        g1_23dof_fast_sac, \
        g1_23dof_fast_sac_v1, \
        unitree_rl_init_g1_23dof_fast_sac_v1, \
        unitree_rl_init_g1_23dof_fast_sac_v2, \
        unitree_rl_init_g1_23dof_fast_sac_v3


from holosoma.config_values.loco.g1.dof23_experiment import \
        g123dof_loc_fastsac_v1


from holosoma.config_values.loco.t1.experiment import t1_29dof, \
        t1_29dof_fast_sac

from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
)

from holosoma.config_values.wbt.g1.dof23_experiment import (
    g1_23dof_wbt_fast_sac,
    g1_23dof_wbt_fast_sac_w_object,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "g1_23dof_fast_sac": g1_23dof_fast_sac,
    "g1_23dof_fast_sac_v1": g1_23dof_fast_sac_v1,
    "unitree_rl_init_g1_23dof_fast_sac_v1": unitree_rl_init_g1_23dof_fast_sac_v1,
    "unitree_rl_init_g1_23dof_fast_sac_v2": unitree_rl_init_g1_23dof_fast_sac_v2,
    "unitree_rl_init_g1_23dof_fast_sac_v3": unitree_rl_init_g1_23dof_fast_sac_v3,
    "g123dof_loc_fastsac_v1": g123dof_loc_fastsac_v1,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,

    "g1_23dof_wbt_fast_sac": g1_23dof_wbt_fast_sac,
    "g1_23dof_wbt_fast_sac_w_object": g1_23dof_wbt_fast_sac_w_object,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
