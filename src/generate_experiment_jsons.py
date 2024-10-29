import json

runs_per_experiment = 50
max_fitness_evaluations = 5000

default_params = {
    "--pop-size": str(50),
    "--max-depth": str(6),
    "--min-depth": str(0),
    "--p-cross": str(0.5),
    "--p-mut": str(0.5),
    "--crossover-operator": "cxOnePoint",
    "--mutation-operator": "mutUniform",
    "--tournament-size": str(7),
    #"--model-weights": "../experiments/model_training/dropout_influence/0/505_tecator/2024-06-10_15-21-16_dhcpf118.fit.vutbr.cz/model.pth",
    "--model-weights": "../model-tecator-big.pth",
    "--model-dataset": "505_tecator",
    "--mut-param": str(0.005),
    "--mut-ratio-param": str(0.1),
    "--dataset": "505_tecator",
    "--mutation-force-change": False,
    "--mutation-max-trials": str(3)
}


def generate_variant(variant_name, variants, non_default_params=None):
    params = default_params.copy()
    if non_default_params is not None:
        params.update(non_default_params)
    experiments = []



    for variant in variants:
        if variant_name == "--pop-size":
            params["--generations"] = str(max_fitness_evaluations // variant)
        elif variant_name == "--generations":
            params["--pop-size"] = str(max_fitness_evaluations // variant)
        else:
            params["--generations"] = str(max_fitness_evaluations // int(params["--pop-size"]))

        params[variant_name] = str(variant)
        force_diff_string = f"--force{params["--mutation-force-change"]}-trials{params["--mutation-max-trials"]}"\
            if params["--mutation-force-change"] else ""
        params["--output-dir"] = (f"../runs/evolution/{params["--mutation-operator"]}_{params["--crossover-operator"]}"
                                  f"/{variant_name}{force_diff_string}/{params["--dataset"]}/{variant_name}-{variant}/")
        command = {key: value for key, value in params.items()}

        for key, value in command.items():
            if type(value) is bool:
                if value:
                    command[key] = ""
                else:
                    command[f"--no-{key[:2]}"] = ""
            else:
                command[key] = value

        experiments.append({
            "command": [item for pair in params.items() for item in pair if type(item) is not bool],
            "remaining-runs": runs_per_experiment,
            "requires-gpu": False,
            "total-runs": runs_per_experiment
        })
    return experiments


def benchmark_de_mut_cxOnePoint():
    experiments = []
    de_mut_cxOnePoint_defaults = {
        "--p-mut": "0.85",
        "--p-cross": "0",
        "--mut-ratio-param": "0.05",
        "--mutation-operator": "de_mut",
        "--mut-param": "0.75",
        "--pop-size": "100"
    }
    # experiments.extend(generate_variant("--mut-param", [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))
    # experiments.extend(generate_variant("--mut-ratio-param", [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))

    # experiments.extend(generate_variant("--pop-size", [10, 50, 100, 200, 500, 1000],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))

    experiments.extend(generate_variant("--p-mut",
                                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        non_default_params=de_mut_cxOnePoint_defaults))

    # experiments.extend(generate_variant("--p-mut", [0.8],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))
    #
    # experiments.extend(generate_variant("--p-cross", [0, 0.1, 0.25, 0.75, 0.9, 1],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))
    # experiments.extend(generate_variant("--tournament-size", [5, 10],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))

    # experiments.extend(generate_variant("--dataset", ["228_elusage"],
    #                                     non_default_params=de_mut_cxOnePoint_defaults))

    return experiments


def benchmark_mutUniform_cxOnePoint():
    mutUniform_cxOnePoint_defaults = {
        "--pop-size": str(50),
        "--p-cross": str(0),
        "--p-mut": str(1.0),
        "--crossover-operator": "cxOnePoint",
        "--mutation-operator": "mutUniform",
        "--tournament-size": str(10),
    }

    experiments = []
    # experiments.extend(generate_variant("--pop-size", [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2500],
    #                                     non_default_params=mutUniform_cxOnePoint_defaults))

    # experiments.extend(generate_variant("--p-mut",
    #                                     [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #                                     non_default_params=mutUniform_cxOnePoint_defaults))
    experiments.extend(generate_variant("--tournament-size",
                                        [1, 2, 5, 7, 10, 15, 20, 30, 50],
                                        non_default_params=mutUniform_cxOnePoint_defaults))

    return experiments


def benchmark_mut_rev_cosine_dist():
    experiments = []
    mut_rev_cosine_dist_defaults = {
        "--p-mut": "0.85",
        "--p-cross": "0",
        "--mut-ratio-param": "0.1",
        "--mutation-operator": "mut_rev_cosine_dist",
        "--mut-param": "0.8",
        "--pop-size": "100"
    }
    experiments.extend(generate_variant("--mut-param",
                                        [0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.5, 2],
                                        non_default_params=mut_rev_cosine_dist_defaults))
    return experiments


def benchmark_add_random_noise():
    experiments = []
    mut_rev_cosine_dist_defaults = {
        "--p-mut": "0.9",
        "--p-cross": "0",
        "--mut-ratio-param": "0.05",
        "--mutation-operator": "mut_add_random_noise_gaussian",
        "--mut-param": "0.5",
        "--pop-size": "200",
        "--mutation-force-change": True,
        "--mutation-max-trials": str(20)
    }
    experiments.extend(generate_variant("--mut-param",
                                        [0.01, 0.05, 0.1, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0],
                                        non_default_params=mut_rev_cosine_dist_defaults))
    # experiments.extend(generate_variant("--mut-ratio-param",
    #                                     [0.02, 0.03, 0.04],
    #                                     non_default_params=mut_rev_cosine_dist_defaults))
    # experiments.extend(generate_variant("--pop-size",
    #                                     [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2500],
    #                                     non_default_params=mut_rev_cosine_dist_defaults))
    # experiments.extend(generate_variant("--p-mut",
    #                                     [0],
    #                                     non_default_params=mut_rev_cosine_dist_defaults))
    return experiments


def benchmark_mut_rev_euclid_dist():
    experiments = []
    mut_rev_euclid_dist_defaults = {
        "--p-mut": "0.85",
        "--p-cross": "0",
        "--mutation-operator": "mut_rev_euclid_dist",
        "--mut-param": "10",
        "--pop-size": "100",
        "--mutation-force-change": True,
        "--mutation-max-trials": str(20)
    }
    experiments.extend(generate_variant("--mut-param",
                                        [1, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
                                        non_default_params=mut_rev_euclid_dist_defaults))
    # experiments.extend(generate_variant("--pop-size",
    #                                     [100, 200, 500, 1000, 2500],
    #                                     non_default_params=mut_rev_euclid_dist_defaults))
    # experiments.extend(generate_variant("--p-mut",
    #                                     [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #                                     non_default_params=mut_rev_euclid_dist_defaults))
    return experiments



experiments = []
# experiments.extend(generate_variant("--pop-size", [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
# experiments.extend(generate_variant("--p-cross", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant("--p-mut", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant("--tournament-size", [1, 2, 3, 5, 7, 10, 20, 50]))

# experiments.extend(generate_variant("--mut-param", [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
#                                    non_default_params={"--mutation-operator": "mut_rev_cosine_dist"}))
#experiments.extend(generate_variant("--generations", [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]))

#experiments.extend(generate_variant("--pop-size", [50]))

# experiments.extend(generate_variant("--mut-param", [0.0001, 0.00001],
#                                     non_default_params={"--mutation-operator": "mut_rev_cosine_dist"}))

# experiments.extend(generate_variant("--mut-param", [5],
#                                     non_default_params={"--mutation-operator": "mut_add_random_noise_gaussian"}))
# experiments.extend(generate_variant("--noise-mut-ration", [0.005, 0.01, 0.05, 0.2, 0.5],
#                                     non_default_params={
#                                         "--mutation-operator": "mut_add_random_noise_gaussian",
#                                         "--mut-param": str(1.25)
#                                     }))
# experiments.extend(generate_variant("--pop-size", [100, 25],
#                                     non_default_params={
#                                         "--mutation-operator": "mut_add_random_noise_gaussian",
#                                         "--mut-param": str(1.25),
#                                         "--mut-ratio-param": "0.2"
#                                     }))


experiments.extend(benchmark_mut_rev_euclid_dist())


with open("../new_expr.json", "w") as experiments_f:
    experiments_f.write(json.dumps(experiments, indent=4, sort_keys=True))
