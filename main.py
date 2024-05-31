import operator
import math
import random

import numpy as np

from functools import partial

import pmlb
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import varAnd

from pmlb import fetch_data
from sym_reg_tree import SymRegTree, get_mapping


def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return 1


toolbox = base.Toolbox()


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),


def eval_symb_reg_pmlb(individual, inputs, targets):
    func = toolbox.compile(expr=individual)
    #data = fetch_data('adult')
    #inputs = data.drop('target', axis=1)
    #targets = data['target']

    outputs = inputs.apply(lambda x: func(*x), axis=1)# , engine="numba")
    return ((outputs - targets) ** 2).sum() / len(targets),


def generate_random_trees(n_trees):
    trees = [toolbox.individual() for _ in range(n_trees)]
    return trees


def eaSimple_with_population_log(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, trees=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        if gen > 5 and trees is not None:
            for ind in population:
                trees.append(ind)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    random.seed(318)

    dataset_name = '4544_GeographicalOriginalofMusic'
    dataset = pmlb.fetch_data(dataset_name)

    pset = gp.PrimitiveSet("MAIN", len(dataset.drop('target', axis=1).columns))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(sqrt, 1)
    for i, val in enumerate(np.linspace(-1, 1, num=21, endpoint=True)):
        pset.addTerminal(val)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset

    print(f"Mapping len: {len(get_mapping(toolbox.pset, ["PAD", "UNKNOWN"]))}")

    toolbox.register("evaluate", eval_symb_reg_pmlb, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    pop = toolbox.population(n=900)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    trees = generate_random_trees(100000)
    pop, log = eaSimple_with_population_log(pop, toolbox, 0.8, 0.05, 55, stats=mstats,
                                            halloffame=hof, verbose=True, trees=trees)
    tokenized_trees = [tree.tokenize(toolbox.pset, 5) for tree in trees]
    trees_array = np.array(tokenized_trees, dtype=int)
    np.random.shuffle(trees_array)
    print(f"Trees shape: {trees_array.shape}")
    np.savetxt("geomusic_dataset_mdepth5.txt", trees_array)

    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()

