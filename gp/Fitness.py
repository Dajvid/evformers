

def eval_symb_reg_pmlb(individual, inputs, targets):
    #func = toolbox.compile(expr=individual)
    func = individual.compile()
    #data = fetch_data('adult')
    #inputs = data.drop('target', axis=1)
    #targets = data['target']

    outputs = inputs.apply(lambda x: func(*x), axis=1)# , engine="numba")
    return ((outputs - targets) ** 2).sum() / len(targets),
