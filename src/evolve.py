import argparse
import operator
import os.path
from collections import namedtuple

import numpy as np
import pandas as pd
import pmlb
import torch

from deap import creator, base, gp, tools

from gp.custom_ea_simple import eaSimple
from gp.Fitness import eval_symb_reg_pmlb, binary_regression_fitness
from gp.Pset import create_basic_symreg_pset, create_basic_logic_pset
from gp.sym_reg_tree import SymRegTree, get_mapping
from gp.mutations import mut_add_random_noise_gaussian, mut_rev_cosine_dist, mut_rev_euclid_dist, de_mut

import warnings

from gpformer.model import Transformer
from gp.crossovers import cross_average_dif, cross_half_half
from gp.Generate_data import even_parity_truth_table

warnings.filterwarnings("ignore")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument("--p-cross", type=float, default=0.5)
    parser.add_argument("--p-mut", type=float, default=0.5)
    parser.add_argument("--crossover-operator", type=str, default="cxOnePoint")
    parser.add_argument("--mutation-operator", type=str, default="mutUniform")
    parser.add_argument("--tournament-size", type=int, default=7)
    parser.add_argument("--dataset", type=str, default="505_tecator")
    parser.add_argument("--output-dir", type=str, default="../runs/evolution")
    parser.add_argument("--verbose", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--model-weights", type=str, default="../model-tecator-0-6-SOT.pth")
    parser.add_argument("--mut-param", type=float, default=0.05)
    parser.add_argument("--mut-ratio-param", type=float, default=0.1)
    parser.add_argument("--model-dataset", type=str, default="505_tecator")
    parser.add_argument("--mutation-force-change", type=bool, action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--mutation-max-trials", type=int, default=3)

    args = parser.parse_args(argv)
    if args.mutation_operator not in ["mutUniform", "mut_add_random_noise_gaussian", "mut_rev_cosine_dist",
                                      "mut_rev_euclid_dist", "de_mut"]:
        raise ValueError(f"Unknown mutation operator: {args.mutation_operator}")
    if args.crossover_operator not in ["cxOnePoint", "cxAverage", "cxHalfHalf"]:
        raise ValueError(f"Unknown crossover operator: {args.crossover_operator}")

    return args


def handle_mut_operator(toolbox, args, pset, pop, stats, model, mapping):
    if args.mutation_operator == "mutUniform":
        toolbox.register("expr_mut", gp.genFull, min_=args.min_depth, max_=args.max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    elif args.mutation_operator == "mut_add_random_noise_gaussian":
        toolbox.register("mutate", mut_add_random_noise_gaussian, pset=pset, mapping=mapping, stats=stats,
                         force_diffent=args.mutation_force_change, max_trials=args.mutation_max_trials,
                         max_depth=args.max_depth, model=model, scaler=args.mut_param, ratio=args.mut_ratio_param)
    elif args.mutation_operator == "mut_rev_cosine_dist":
        toolbox.register("mutate", mut_rev_cosine_dist, pset=pset, stats=stats,
                         force_diffent=args.mutation_force_change, max_trials=args.mutation_max_trials,
                         max_depth=args.max_depth, model=model, mapping=mapping, distance=args.mut_param)
    elif args.mutation_operator == "mut_rev_euclid_dist":
        toolbox.register("mutate", mut_rev_euclid_dist, pset=pset, stats=stats,
                         force_diffent=args.mutation_force_change, max_trials=args.mutation_max_trials,
                         max_depth=args.max_depth, model=model, mapping=mapping, distance=args.mut_param)
    elif args.mutation_operator == "de_mut":
        toolbox.register("mutate", de_mut, pset=pset, model=model, mapping=mapping, stats=stats,
                         force_diffent=args.mutation_force_change, max_trials=args.mutation_max_trials,
                         max_depth=args.max_depth, population=pop, F=args.mut_param, CR=args.mut_ratio_param)
    else:
        raise ValueError(f"Unknown mutation operator: {args.mutation_operator}")

    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=args.max_depth))


def handle_cross_operator(toolbox, args, pset, stats, model, mapping):
    if args.crossover_operator == "cxOnePoint":
        toolbox.register("mate", gp.cxOnePoint)
    elif args.crossover_operator == "cxAverage":
        toolbox.register("mate", cross_average_dif, pset=pset, mapping=mapping, model=model,
                         max_depth=args.max_depth, stats=stats)
    elif args.crossover_operator == "cxHalfHalf":
        toolbox.register("mate", cross_half_half, pset=pset, mapping=mapping, model=model,
                            max_depth=args.max_depth, stats=stats)
    else:
        raise ValueError(f"Unknown crossover operator: {args.crossover_operator}")

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                            max_value=args.max_depth))


def main(argv=None):
    args = parse_args(argv)

    mut_stats = {name: 0 for name in ["success", "trials", "fallbacked", "called", "unchanged"]}
    cross_stats = {name: 0 for name in ["success", "trials", "fallbacked", "called", "unchanged"]}

    try:
        if args.dataset.split('-')[1] == "parity":
            dataset = even_parity_truth_table(int(args.dataset.split('-')[0]))
            pset = create_basic_logic_pset(dataset)
            fitness = binary_regression_fitness
        else:
            dataset = pmlb.fetch_data(args.dataset, local_cache_dir="../datasets/pmlb_cache", dropna=True)
            pset = create_basic_symreg_pset(dataset)
            fitness = eval_symb_reg_pmlb
    except IndexError:
        dataset = pmlb.fetch_data(args.dataset, local_cache_dir="../datasets/pmlb_cache", dropna=True)
        pset = create_basic_symreg_pset(dataset)
        fitness = eval_symb_reg_pmlb

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=args.min_depth, max_=args.max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset
    toolbox.register("evaluate", fitness, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    pop = toolbox.population(n=args.pop_size)

    mapping = get_mapping(pset, ["PAD", "UNKNOWN", "SOT"])
    if (args.mutation_operator in ["de_mut", "mut_add_random_noise_gaussian",
                                  "mut_rev_cosine_dist", "mut_rev_euclid_dist"]
            or args.crossover_operator in ["cxAverage", "cxHalfHalf"]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            if args.dataset.split('-')[1] == "parity":
                model_dataset = even_parity_truth_table(int(args.dataset.split('-')[0]))
                model_pset = create_basic_logic_pset(model_dataset)
            else:
                model_dataset = pmlb.fetch_data(args.model_dataset, local_cache_dir="../datasets/pmlb_cache")
                model_pset = create_basic_symreg_pset(model_dataset)
        except IndexError:
                model_dataset = pmlb.fetch_data(args.model_dataset, local_cache_dir="../datasets/pmlb_cache")
                model_pset = create_basic_symreg_pset(model_dataset)
        model_mapping = get_mapping(model_pset, ["PAD", "UNKNOWN", "SOT"])
        model = Transformer(model_mapping, 2 * args.max_depth, 2, 1, 1,
                            args.max_depth, 2, ignore_pad=False).to(device)
        model.load_state_dict(torch.load(args.model_weights, map_location=torch.device(device)))
    else:
        model = None

    handle_mut_operator(toolbox, args, pset, pop, mut_stats, model, mapping)
    handle_cross_operator(toolbox, args, pset, cross_stats, model, mapping)

    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("fit-avg", np.mean)
    stats_fit.register("fit-std", np.std)
    stats_fit.register("fit-min", np.min)
    stats_fit.register("fit-max", np.max)

    stats_height = tools.Statistics(lambda ind: ind.height)
    stats_height.register("height-avg", np.mean)
    stats_height.register("height-std", np.std)
    stats_height.register("height-min", np.min)
    stats_height.register("height-max", np.max)
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height)

    pop, log = eaSimple(pop, toolbox, args.p_cross, args.p_mut, args.generations, stats=mstats, halloffame=hof,
                        verbose=args.verbose)

    statistics = pd.DataFrame([{**st1, **st2}for st1, st2 in zip(log.chapters["fitness"], log.chapters["height"])])
    statistics["run_id"] = args.run_id
    statistics["dataset"] = args.dataset
    statistics["pop_size"] = args.pop_size
    statistics["generations"] = args.generations
    statistics["max_depth"] = args.max_depth
    statistics["min_depth"] = args.min_depth
    statistics["p_cross"] = args.p_cross
    statistics["p_mut"] = args.p_mut
    statistics["tournament_size"] = args.tournament_size
    statistics["crossover_operator"] = args.crossover_operator
    statistics["mutation_operator"] = args.mutation_operator
    statistics["output_dir"] = args.output_dir
    statistics["model_weights"] = args.model_weights
    statistics["mut_param"] = args.mut_param
    statistics["mut_ratio_param"] = args.mut_ratio_param
    statistics["mut_called"] = mut_stats["called"]
    statistics["mut_success"] = mut_stats["success"]
    statistics["mut_trials"] = mut_stats["trials"]
    statistics["mut_fallbacked"] = mut_stats["fallbacked"]
    statistics["cross_called"] = cross_stats["called"]
    statistics["cross_success"] = cross_stats["success"]
    statistics["cross_trials"] = cross_stats["trials"]
    statistics["cross_fallbacked"] = cross_stats["fallbacked"]
    statistics["mut_unchanged"] = mut_stats["unchanged"]
    statistics["cross_unchanged"] = cross_stats["unchanged"]

    print("mutation stats", mut_stats)
    print("crossover stats", cross_stats)

    os.makedirs(args.output_dir, exist_ok=True)
    statistics.to_pickle(os.path.join(args.output_dir, f"{args.dataset}_run_{args.run_id}.pkl"))


if __name__ == '__main__':
    main()
