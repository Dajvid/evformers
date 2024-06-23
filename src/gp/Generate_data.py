import operator
import itertools
import numpy as np
import pandas as pd
import pmlb
from deap.algorithms import varAnd
from deap import tools, base, creator, gp
from deap.gp import genHalfAndHalf

from gp.Fitness import eval_symb_reg_pmlb
from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import SymRegTree, get_mapping


def eaSimple_with_population_log(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, trees=None, skip_first_n_generations=5):
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
        print(f"Generation {gen} / {ngen}\r", end="")
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
        if gen > skip_first_n_generations and trees is not None:
            for ind in population:
                trees.append(ind)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def generate_random_trees(pset, n_trees, min_depth=0, max_depth=8):
    trees = []
    for _ in range(n_trees):
        tree = SymRegTree(genHalfAndHalf(pset, min_depth, max_depth))
        tree.pset = pset
        trees.append(tree)

    return trees


def generate_trees_from_evolution(pset, dataset, min_depth=0, max_depth=8, fitness=eval_symb_reg_pmlb,
                                  n_generations=55, n_trees_per_generation=900, skip_first_n_generations=5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin, pset=pset)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset

    toolbox.register("evaluate", fitness, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=min_depth, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    trees = []
    pop = toolbox.population(n=n_trees_per_generation)
    eaSimple_with_population_log(pop, toolbox, 0.8, 0.05, ngen=n_generations,
                                 verbose=False, trees=trees, skip_first_n_generations=skip_first_n_generations)

    return trees


def even_parity_truth_table(number_of_inputs: int) -> pd.DataFrame:
    inputs = list(itertools.product([0, 1], repeat=number_of_inputs))
    outputs = [np.sum(i) % 2 for i in inputs]
    df = pd.DataFrame(inputs, columns=[f'X{i}' for i in range(number_of_inputs)])
    df["target"] = outputs

    return df



