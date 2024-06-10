import json

runs_per_experiment = 100
max_fitness_evaluations = 1000000


def generate_variant_over_datasets(variant_name, variants):
    datasets = ["505_tecator", "588_fri_c4_1000_100", "503_wind", "228_elusage"]
    experiments = []

    params = {
        "--pop-size": 50,
        "--generations": 100,
        "--max-depth": 8,
        "--min-depth": 0,
        "--p-cross": 0.7,
        "--p-mut": 0.3,
        "--crossover_operator": "cxOnePoint",
        "--mutation_operator": "mutUniform",
        "--tournament-size": 3,
        "--verbose": False
    }

    for dataset in datasets:
        for variant in variants:
            params["--dataset"] = dataset
            if variant_name == "--pop-size":
                params["--generations"] = max_fitness_evaluations // variant
            elif variant_name == "--generations":
                params["--pop-size"] = max_fitness_evaluations // variant

            params[variant_name] = variant
            params["output-dir"] = f"../runs/evolution/{dataset}/{variant_name}-{variant}/"
            experiments.append({
                "command": [item for pair in params.items() for item in pair],
                "remaining-runs": runs_per_experiment,
                "requires-gpu": False,
                "total-runs": runs_per_experiment
            })
    return experiments


experiments = []
experiments.extend(generate_variant_over_datasets("--popsize", [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
experiments.extend(generate_variant_over_datasets("--p-cross", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
experiments.extend(generate_variant_over_datasets("--p-mut", [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.95, 1]))
experiments.extend(generate_variant_over_datasets("--tournament-size", [1, 2, 3, 5, 7, 10, 20, 50]))

with open("../new_expr.json", "w") as experiments_f:
    experiments_f.write(json.dumps(experiments, indent=4, sort_keys=True))
