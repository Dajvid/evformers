import json

runs_per_experiment = 50
max_fitness_evaluations = 10000

default_params = {
    "--pop-size": str(50),
    "--generations": str(int(max_fitness_evaluations // 50)),
    "--max-depth": str(6),
    "--min-depth": str(0),
    "--p-cross": str(0.5),
    "--p-mut": str(0.5),
    "--crossover-operator": "cxOnePoint",
    "--mutation-operator": "mutUniform",
    "--tournament-size": str(7),
    "--model-weights": "../experiments/model_training/dropout_influence/0/505_tecator/model.pth",
    "--mut-param": str(0.005)
}


def generate_variant(variant_name, variants, non_default_params=None):
    params = default_params.copy()
    if non_default_params is not None:
        params.update(non_default_params)
    datasets = ["505_tecator"]
    experiments = []



    for dataset in datasets:
        for variant in variants:
            params["--dataset"] = dataset
            if variant_name == "--pop-size":
                params["--generations"] = str(max_fitness_evaluations // variant)
            elif variant_name == "--generations":
                params["--pop-size"] = str(max_fitness_evaluations // variant)

            params[variant_name] = str(variant)
            params["--output-dir"] = (f"../runs/evolution/{params["--mutation-operator"]}_{params["--crossover-operator"]}"
                                      f"/{variant_name}/{dataset}/{variant_name}-{variant}/")
            experiments.append({
                "command": [item for pair in params.items() for item in pair],
                "remaining-runs": runs_per_experiment,
                "requires-gpu": False,
                "total-runs": runs_per_experiment
            })
    return experiments


experiments = []
#experiments.extend(generate_variant("--pop-size", [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
#experiments.extend(generate_variant(default_params, "--p-cross", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant(default_params, "--p-mut", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant(default_params, "--tournament-size", [1, 2, 3, 5, 7, 10, 20, 50]))

experiments.extend(generate_variant("--mut-param", [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                                    non_default_params={"--mutation-operator": "mut_rev_cosine_dist"}))
#experiments.extend(generate_variant("--generations", [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]))

with open("../new_expr.json", "w") as experiments_f:
    experiments_f.write(json.dumps(experiments, indent=4, sort_keys=True))
