import pandas as pd
import pmlb
import torch
from deap import creator, base, gp, tools

from gp.Fitness import eval_symb_reg_pmlb
from gp.Pset import create_basic_symreg_pset
from gp.mutations import mut_add_random_noise_gaussian, mut_rev_cosine_dist, mut_rev_euclid_dist
from gp.sym_reg_tree import SymRegTree, get_mapping
from gpformer.model import Transformer
from gp.crossovers import cross_average_dif

device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = pmlb.fetch_data("505_tecator", local_cache_dir="../datasets/pmlb_cache", dropna=True)
pset = create_basic_symreg_pset(dataset)
fitness = eval_symb_reg_pmlb
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", SymRegTree, pset=pset, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)


mapping = get_mapping(pset, ["PAD", "UNKNOWN", "SOT"])

model = Transformer(mapping, 2 * 6, 2, 1, 1,
                    6, 2, ignore_pad=False).to(device)
model.load_state_dict(torch.load("../model-tecator-big.pth", map_location=torch.device(device)))
max_trials = 20
repetitions = 250

# all_stats = []
# for scaler in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
#     for ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
#         stats = {"called": 0, "trials": 0, "success": 0, "fallbacked": 0, "unchanged": 0}
#         print(f"scaler: {scaler}, ratio: {ratio}")
#         for i in range(repetitions):
#             print(f"{i} / {repetitions}\r", end="")
#             individual = toolbox.individual()
#             mutated = mut_add_random_noise_gaussian(individual, pset, mapping, 6, model, scaler, stats,
#                                                     ratio=ratio, force_diffent=True, max_trials=max_trials)[0]
#         stats["ratio"] = ratio
#         stats["scaler"] = scaler
#         print(stats)
#         all_stats.append(stats)
#
# pd.DataFrame(all_stats).to_pickle(f"../mut_add_random_noise_gaussian_stats_force_diff_trials_{max_trials}.pkl")


all_stats = []
for distance in range(1, 60):
    stats = {"called": 0, "trials": 0, "success": 0, "fallbacked": 0, "unchanged": 0}
    print(f"distance: {distance}")
    for i in range(repetitions):
        print(f"{i} / {repetitions}\r", end="")
        individual = toolbox.individual()
        mutated = mut_rev_euclid_dist(individual, pset, mapping, 6, model, distance, stats,
                                      force_diffent=True, max_trials=max_trials)[0]
        #print(f"{str(individual)} -> {str(mutated)}")
    stats["distance"] = distance
    print(stats)
    all_stats.append(stats)

pd.DataFrame(all_stats).to_pickle(f"../mut_rev_euclid_dist_force_diff_trials_{max_trials}.pkl")


# all_stats = []
# for distance in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
#     stats = {"called": 0, "trials": 0, "success": 0, "fallbacked": 0, "unchanged": 0}
#     print(f"distance: {distance}")
#     for i in range(repetitions):
#         print(f"{i} / {repetitions}\r", end="")
#         individual1 = toolbox.individual()
#         individual2 = toolbox.individual()
#         mutated_1, mutated_2 = cross_average_dif(individual1, individual2, pset, mapping, 6, model, stats)
#         print(f"{str(individual1)} -> {str(mutated_1)}")
#         print(f"{str(individual2)} -> {str(mutated_2)}")
#     stats["distance"] = distance
#     all_stats.append(stats)