import operator
import math
import random

import numpy as np

from functools import partial

import pmlb
from deap import base
from deap import creator
from deap import tools
from deap import gp

from gp.Fitness import eval_symb_reg_pmlb
from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import SymRegTree, get_mapping
from gp.Generate_data import eaSimple_with_population_log


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),


def main():
    random.seed(318)
    dataset = pmlb.fetch_data("4544_GeographicalOriginalofMusic")
    pset = create_basic_symreg_pset(dataset)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset

    print(f"Mapping len: {len(get_mapping(toolbox.pset, ["PAD", "UNKNOWN"]))}")

    toolbox.register("evaluate", eval_symb_reg_pmlb, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

    pop = toolbox.population(n=900)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    trees = []# generate_random_trees(1000000)
    pop, log = eaSimple_with_population_log(pop, toolbox, 0.8, 0.05, 55, stats=mstats,
                                            halloffame=hof, verbose=True, trees=trees)
    avg_depth = sum([tree.height for tree in trees]) / len(trees)
    print("Average depth of trees: ", avg_depth)
    tokenized_trees = [tree.tokenize(toolbox.pset, 4) for tree in trees]
    trees_array = np.array(tokenized_trees, dtype=int)
    np.random.shuffle(trees_array)
    print(f"Trees shape: {trees_array.shape}")
    np.savetxt("geomusic_dataset_depth4-4.txt", trees_array)
    print(f"Percentage of PAD tokens: {np.sum(trees_array == 0) / trees_array.size * 100} %")

    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()

