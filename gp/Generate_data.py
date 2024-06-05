import operator

import pmlb
from deap.algorithms import varAnd
from deap import tools, base, creator, gp
from deap.gp import genHalfAndHalf

from gp.Fitness import eval_symb_reg_pmlb
from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import SymRegTree, get_mapping


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
        if gen > 5 and trees is not None:
            for ind in population:
                trees.append(ind)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def generate_random_trees(pset, n_trees, min_depth=0, max_depth=9):
    trees = [SymRegTree(genHalfAndHalf(pset, min_depth, max_depth)) for _ in range(n_trees)]
    return trees


def generate_trees_from_evolution(pset, dataset, min_height=0, max_height=9,
                                  n_generations=55, n_trees_per_generation=900, skip_first_n_generations=5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", SymRegTree, fitness=creator.FitnessMin, pset=pset)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.pset = pset

    toolbox.register("evaluate", eval_symb_reg_pmlb, inputs=dataset.drop('target', axis=1),
                     targets=dataset['target'])
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=min_height, max_=max_height)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))

    trees = []
    pop = toolbox.population(n=n_trees_per_generation)
    eaSimple_with_population_log(pop, toolbox, 0.8, 0.05, ngen=n_generations,
                                 verbose=False, trees=trees)

    return trees
