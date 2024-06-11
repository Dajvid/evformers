import numpy as np
import torch

from src.gp.sym_reg_tree import SymRegTree


def mut_add_random_noise_gaussian(individual, pset, min_depth, max_depth, model):
    tokenized = individual.tokenize(pset, max_depth)
    encoded = model.encode(tokenized)
    encoded += torch.randn_like(encoded)
    decoded = model.decode(encoded)

    return SymRegTree(pset, decoded)
