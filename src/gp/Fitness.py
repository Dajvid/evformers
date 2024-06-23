

def eval_symb_reg_pmlb(individual, inputs, targets):
    func = individual.compile()

    outputs = inputs.apply(lambda x: func(*x), axis=1)# , engine="numba")
    return ((outputs - targets) ** 2).sum() / len(targets),


def binary_regression_fitness(individual, inputs, targets):
    func = individual.compile()

    outputs = inputs.apply(lambda x: func(*x), axis=1)
    return (outputs.astype("bool") != targets).sum(),
