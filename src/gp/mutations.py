import numpy as np
import torch
from deap import gp
from deap.gp import mutUniform
from functools import partial

from gp.sym_reg_tree import SymRegTree

device = "cuda" if torch.cuda.is_available() else "cpu"


def mut_add_random_noise_gaussian(individual, pset, mapping, max_depth, model, scaler, ratio=0.1):
    tokenized = individual.tokenize(max_depth, mapping, add_SOT=True)
    encoded = model.encode(torch.tensor(tokenized).to(device))
    trials = 0
    mutated = None

    while mutated is None and trials < 3:
        trials += 1
        mask = (torch.randn_like(encoded) < 0.1).float()
        noise = torch.randn_like(encoded) * mask * scaler
        decoded = model.decode(encoded + noise)

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
        except (RuntimeError, IndexError):
            mutated = None

    if mutated is None:
        # fallback to mutuniform if no valid mutation is found in 3 trials
        mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]

    return mutated,


def mut_rev_cosine_dist(individual, pset, mapping, max_depth, model, distance):
    def gram_schmidt_orthogonalization(A, random_vector):
        # Project the random vector onto A
        projection = torch.dot(random_vector, A) / torch.dot(A, A) * A
        orthogonal_vector = random_vector - projection

        # Normalize the orthogonal vector to make it a unit vector
        orthogonal_unit_vector = orthogonal_vector / torch.norm(orthogonal_vector)
        return orthogonal_unit_vector

    def random_orthogonal_unit_vector(A):
        while True:
            # Generate a random vector
            random_vector = torch.randn_like(A)
            orthogonal_vector = gram_schmidt_orthogonalization(A, random_vector)

            if torch.norm(orthogonal_vector) > 1e-6:  # Avoid degenerate case where the orthogonal vector is too small
                return orthogonal_vector

    def reverse_cosine_distance(A, D):
        if D < 0 or D > 2:
            raise ValueError("Cosine distance must be in the range [0, 2].")

        A = A / torch.norm(A)  # Ensure A is a unit vector
        cos_theta = 1 - D
        cos_theta = torch.tensor(cos_theta, dtype=torch.float32).to(device)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)

        U = random_orthogonal_unit_vector(A)

        B = cos_theta * A + sin_theta * U
        return B


    tokenized = individual.tokenize(max_depth, mapping, add_SOT=True)
    encoded = model.encode(torch.tensor(tokenized).to(device))
    trials = 0
    mutated = None

    while mutated is None and trials < 3:
        trials += 1
        original_shape = encoded.shape
        distanced = reverse_cosine_distance(encoded.flatten(), distance)
        decoded = model.decode(distanced.reshape(original_shape))

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
        except (RuntimeError, IndexError):
            mutated = None

    if mutated is None:
        # fallback to mutuniform if no valid mutation is found in 3 trials
        mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]

    return mutated,
