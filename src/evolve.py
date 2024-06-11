import argparse
import operator
import os.path

import numpy as np
import pandas as pd
import pmlb

from deap import creator, base, gp, tools
from deap.algorithms import eaSimple

from gp.Fitness import eval_symb_reg_pmlb
from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import SymRegTree

import warnings
warnings.filterwarnings("ignore")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument("--p-cross", type=float, default=0.7)
    parser.add_argument("--p-mut", type=float, default=0.3)
    parser.add_argument("--crossover-operator", type=str)
    parser.add_argument("--mutation-operator", type=str)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="505_tecator")
    parser.add_argument("--output-dir", type=str, default="../runs/evolution")
    parser.add_argument("--verbose", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-id", type=int, default=0)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset = pmlb.fetch_data(args.dataset, local_cache_dir="../datasets/pmlb_cache")
    pset = create_basic_symreg_pset(dataset)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=args.min_depth, max_=args.max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset
    toolbox.register("evaluate", eval_symb_reg_pmlb, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=args.min_depth, max_=args.max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                            max_value=args.max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=args.max_depth))

    pop = toolbox.population(n=args.pop_size)
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
    #mstats = tools.Statistics(lambda ind: ind.fitness.values)

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

    os.makedirs(args.output_dir, exist_ok=True)
    statistics.to_pickle(os.path.join(args.output_dir, f"{args.dataset}_run_{args.run_id}.pkl"))


if __name__ == '__main__':
    main()
