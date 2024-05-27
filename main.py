import operator
import math
import random

import numpy

from functools import partial

import pmlb
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from pmlb import fetch_data
from sym_reg_tree import SymRegTree

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
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


def main():
    random.seed(318)

    dataset_name = '4544_GeographicalOriginalofMusic'
    dataset = pmlb.fetch_data(dataset_name)

    pset = gp.PrimitiveSet("MAIN", len(dataset.drop('target', axis=1).columns))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
    #pset.renameArguments(ARG0='x')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    #toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
    toolbox.register("evaluate", eval_symb_reg_pmlb, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=20)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))

    a = toolbox.individual()
    a.tokenize(pset, 5)

    pop = toolbox.population(n=900)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.05, 5000, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()

