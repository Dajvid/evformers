import math
import operator
import pandas as pd
import numpy as np

from deap import gp
from sympy import Or, And, Not


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


def create_basic_symreg_pset(dataset: pd.DataFrame) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", len(dataset.drop('target', axis=1).columns))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(sqrt, 1)
    for i, val in enumerate(np.linspace(-1, 1, num=21, endpoint=True)):
        pset.addTerminal(val)

    return pset


def create_basic_logic_pset(dataset: pd.DataFrame) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", len(dataset.drop('target', axis=1).columns))
    pset.addPrimitive(And, 2)
    pset.addPrimitive(Or, 2)
    pset.addPrimitive(Not, 1)
    for i, val in enumerate([True, False]):
        pset.addTerminal(val)

    return pset
