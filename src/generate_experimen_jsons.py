import json

runs_per_experiment = 100
max_fitness_evaluations = 30000


def generate_variant_over_datasets(variant_name, variants):
    #datasets = ["505_tecator", "588_fri_c4_1000_100", "503_wind", "228_elusage"]
    datasets = ["505_tecator"]
    experiments = []

    params = {
        "--pop-size": str(50),
        "--generations": str(100),
        "--max-depth": str(8),
        "--min-depth": str(0),
        "--p-cross": str(0.7),
        "--p-mut": str(0.3),
        "--crossover-operator": "cxOnePoint",
        "--mutation-operator": "mutUniform",
        "--tournament-size": str(3)
    }

    for dataset in datasets:
        for variant in variants:
            params["--dataset"] = dataset
            if variant_name == "--pop-size":
                params["--generations"] = str(max_fitness_evaluations // variant)
            elif variant_name == "--generations":
                params["--pop-size"] = str(max_fitness_evaluations // variant)

            params[variant_name] = str(variant)
            params["--output-dir"] = f"../runs/evolution/{variant_name}/{dataset}/{variant_name}-{variant}/"
            experiments.append({
                "command": [item for pair in params.items() for item in pair],
                "remaining-runs": runs_per_experiment,
                "requires-gpu": False,
                "total-runs": runs_per_experiment
            })
    return experiments


experiments = []
#experiments.extend(generate_variant_over_datasets("--pop-size", [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
experiments.extend(generate_variant_over_datasets("--p-cross", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant_over_datasets("--p-mut", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
# experiments.extend(generate_variant_over_datasets("--tournament-size", [1, 2, 3, 5, 7, 10, 20, 50]))

with open("../new_expr.json", "w") as experiments_f:
    experiments_f.write(json.dumps(experiments, indent=4, sort_keys=True))
