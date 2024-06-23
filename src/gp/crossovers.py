import torch
from deap.gp import cxOnePoint

from gp.sym_reg_tree import SymRegTree

device = "cuda" if torch.cuda.is_available() else "cpu"


def cross_average_dif(ind1, ind2, pset, mapping, max_depth, model, stats):
    ind1_encoded = ind1.embedding(model, device, max_depth, mapping)
    ind2_encoded = ind2.embedding(model, device, max_depth, mapping)
    stats["called"] += 1

    mutated_1 = SymRegTree.from_tokenized_tree(model.decode((ind1_encoded + ind2_encoded) / 2), mapping, pset)
    mutated_2 = SymRegTree.from_tokenized_tree(model.decode((ind1_encoded - ind2_encoded)), mapping, pset)
    try:
        stats["trials"] += 1
        mutated_1.padded = mutated_1.add_padding(pset, max_depth)
        mutated_1.fitness = ind1.fitness
        mutated_1.pset = ind1.pset
        stats["success"] += 1
    except (RuntimeError, IndexError):
        stats["fallbacked"] += 1
        mutated_1, mutated_2 = ind1, ind2

    try:
        stats["trials"] += 1
        mutated_2.padded = mutated_2.add_padding(pset, max_depth)
        mutated_2.fitness = ind1.fitness
        mutated_2.pset = ind1.pset
        stats["success"] += 1
    except (RuntimeError, IndexError):
        stats["fallbacked"] += 1
        mutated_2 = ind2

    return mutated_1, mutated_2


def cross_half_half(ind1, ind2, pset, mapping, max_depth, model, stats):
    ind1_encoded = ind1.embedding(model, device, max_depth, mapping)
    ind2_encoded = ind2.embedding(model, device, max_depth, mapping)
    stats["called"] += 1


    mask = torch.rand_like(ind1_encoded) < 0.5
    mutated_1 = SymRegTree.from_tokenized_tree(model.decode(torch.where(mask, ind1_encoded, ind2_encoded)),
                                              mapping, pset)
    mutated_2 = SymRegTree.from_tokenized_tree(model.decode(torch.where(mask, ind2_encoded, ind1_encoded)),
                                              mapping, pset)

    try:
        stats["trials"] += 1
        mutated_1.padded = mutated_1.add_padding(pset, max_depth)
        mutated_1.fitness = ind1.fitness
        mutated_1.pset = ind1.pset
        stats["success"] += 1
    except (RuntimeError, IndexError):
        stats["fallbacked"] += 1
        mutated_1, mutated_2 = cxOnePoint(ind1, ind2)

    try:
        stats["trials"] += 1
        mutated_2.padded = mutated_2.add_padding(pset, max_depth)
        mutated_2.fitness = ind1.fitness
        mutated_2.pset = ind1.pset
        stats["success"] += 1
    except (RuntimeError, IndexError):
        stats["fallbacked"] += 1
        _, mutated_2 = cxOnePoint(ind1, ind2)

    return mutated_1, mutated_2
