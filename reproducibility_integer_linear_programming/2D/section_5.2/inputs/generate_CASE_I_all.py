"""Generate all config files for running CASE I examples"""


import os
import json

import numpy as np

# load base common to all cases 
with open("base_config_ufsc.json", "r") as f:
    model = json.load(f)

# directory where results will be stored
outdir = "CASE_I"

# set penalization exponent
model["material"]["salt_penal"] = 2
model["material"]["penal"] = 1

# dictionary for cases b, d, g, l (alternatively, cases 1, 2, 3, 4)
cases = {
        "b": {
            "num_sources": 1,
            "num_receivers": 21,
            "source_pos": [[1.95, src] for src in np.linspace(0.049, 0.051, 1)],
            "receiver_locations": [[depth, rec] for rec in np.linspace(0.05, 1.95, 21) for depth in [1.95]],
            "water": {"depth_top": 1.85}
            },    
        "d": {
            "num_sources": 3,
            "num_receivers": 21,
            "source_pos": [[1.95, src] for src in np.linspace(0.05, 1.95, 3)],
            "receiver_locations": [[depth, rec] for rec in np.linspace(0.05, 1.95, 21) for depth in [1.95]],
            "water": {"depth_top": 1.85}
            },    
        "g": {
            "num_sources": 2,
            "num_receivers": 21,
            "source_pos": [[1.95, src] for src in np.linspace(0.05, 1.95, 2)],
            "receiver_locations": [[depth, rec] for rec in np.linspace(0.05, 1.95, 21) for depth in [0.05]],
            "water": {"depth_bottom": 0.15}
            },    
        "l": {
            "num_sources": 3,
            "num_receivers": 42,
            "source_pos": [[1.95, src] for src in np.linspace(0.05, 1.95, 3)],
            "receiver_locations": [[depth, rec] for rec in np.linspace(0.05, 1.95, 21) for depth in [0.05, 1.95]],
            "water": {"depth_top": 1.85, "depth_bottom": 0.15}
            }    
}
# parallelism params
model["parallelism"] = {"type": "custom", "num_cores_per_shot": 1}
# acquisition params
model["acquisition"] = {}
model["acquisition"]["source_type"] = "Ricker"
model["acquisition"]["amplitude"] = 10
model["acquisition"]["frequency"] = 2
model["acquisition"]["delay"] = 1
# water params
# model["water"] = {"depth_top": 1.85, "depth_bottom": 0.15}
# model["water"] = {"depth_top": 1.85}
# cplex params
model["cplex"] = {
        "gbar": 1,
        "mul_beta": 25,
        "use_rmin": True,
        "mul_rmin": 30,
        "lim_rmin": 50,
        "epsilons": 0.02,
        "amin": 0.01,
        "amax": 1
}

# BATCH 1
####################
for beta in [0.01, 0.04, 0.07, 0.1]:
    for gamma in [0.5, 0.999]:
        for radius in [0.1, 0.4, 0.7, 1.0]:
            model["cplex"]["beta"] = beta
            model["cplex"]["gamma_m"] = gamma
            model["cplex"]["gamma_v"] = gamma
            model["cplex"]["rmin"] = radius
            for case in cases:
                model["acquisition"]["num_sources"] = cases[case]["num_sources"]
                model["acquisition"]["num_receivers"] = cases[case]["num_receivers"]
                model["acquisition"]["source_pos"] = cases[case]["source_pos"]
                model["acquisition"]["receiver_locations"] = cases[case]["receiver_locations"]
                model["water"] = cases[case]["water"]
                config_file = ("ls_simple"
                              + '_' + str(case)
                              + '_penal_' + str(model["material"]["salt_penal"])
                              + '_amin_' + str(model["cplex"]["amin"])
                              + '_radius_' + str(radius)
                              + '_beta_' + str(beta)
                              + '_gamma_' + str(gamma) + '.json'
                              )
                # save case
                with open(os.path.join(outdir, "batch1", config_file), "w") as f:
                    json.dump(model, f, indent=4)
                

# BATCH 2
####################
for beta in [0.001, 0.005, 0.01, 0.02]:
    for gamma in [ 0.9999]:
        for radius in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            model["cplex"]["beta"] = beta
            model["cplex"]["gamma_m"] = gamma
            model["cplex"]["gamma_v"] = gamma
            model["cplex"]["rmin"] = radius
            for case in cases:
                model["acquisition"]["num_sources"] = cases[case]["num_sources"]
                model["acquisition"]["num_receivers"] = cases[case]["num_receivers"]
                model["acquisition"]["source_pos"] = cases[case]["source_pos"]
                model["acquisition"]["receiver_locations"] = cases[case]["receiver_locations"]
                model["water"] = cases[case]["water"]
                config_file = ("ls_simple"
                              + '_' + str(case)
                              + '_penal_' + str(model["material"]["salt_penal"])
                              + '_amin_' + str(model["cplex"]["amin"])
                              + '_radius_' + str(radius)
                              + '_beta_' + str(beta)
                              + '_gamma_' + str(gamma) + '.json'
                              )
                # save case
                with open(os.path.join(outdir, "batch2", config_file), "w") as f:
                    json.dump(model, f, indent=4)

# BATCH 3
####################
for beta in [0.01, 0.04, 0.07, 0.1]:
    for gamma in [0.9, 0.99]:
        for radius in [0.1, 0.4, 0.7, 1.0]:
            model["cplex"]["beta"] = beta
            model["cplex"]["gamma_m"] = gamma
            model["cplex"]["gamma_v"] = gamma
            model["cplex"]["rmin"] = radius
            for case in cases:
                model["acquisition"]["num_sources"] = cases[case]["num_sources"]
                model["acquisition"]["num_receivers"] = cases[case]["num_receivers"]
                model["acquisition"]["source_pos"] = cases[case]["source_pos"]
                model["acquisition"]["receiver_locations"] = cases[case]["receiver_locations"]
                model["water"] = cases[case]["water"]
                config_file = ("ls_simple"
                              + '_' + str(case)
                              + '_penal_' + str(model["material"]["salt_penal"])
                              + '_amin_' + str(model["cplex"]["amin"])
                              + '_radius_' + str(radius)
                              + '_beta_' + str(beta)
                              + '_gamma_' + str(gamma) + '.json'
                              )
                # save case
                with open(os.path.join(outdir, "batch3", config_file), "w") as f:
                    json.dump(model, f, indent=4)

# BATCH 4
####################
for beta in [0.001, 0.005, 0.01, 0.02]:
    for gamma in [ 0.999]:
        for radius in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            model["cplex"]["beta"] = beta
            model["cplex"]["gamma_m"] = gamma
            model["cplex"]["gamma_v"] = gamma
            model["cplex"]["rmin"] = radius
            for case in cases:
                model["acquisition"]["num_sources"] = cases[case]["num_sources"]
                model["acquisition"]["num_receivers"] = cases[case]["num_receivers"]
                model["acquisition"]["source_pos"] = cases[case]["source_pos"]
                model["acquisition"]["receiver_locations"] = cases[case]["receiver_locations"]
                model["water"] = cases[case]["water"]
                config_file = ("ls_simple"
                              + '_' + str(case)
                              + '_penal_' + str(model["material"]["salt_penal"])
                              + '_amin_' + str(model["cplex"]["amin"])
                              + '_radius_' + str(radius)
                              + '_beta_' + str(beta)
                              + '_gamma_' + str(gamma) + '.json'
                              )
                # save case
                with open(os.path.join(outdir, "batch4", config_file), "w") as f:
                    json.dump(model, f, indent=4)

