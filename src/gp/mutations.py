import random

import numpy as np
import torch
from deap import gp
from deap.gp import mutUniform
from functools import partial
import torch.nn.functional as F
from gp.sym_reg_tree import SymRegTree

device = "cuda" if torch.cuda.is_available() else "cpu"


def mut_add_random_noise_gaussian(individual, pset, mapping, max_depth, model, scaler, stats, ratio=0.1,
                                  force_diffent=False, max_trials=3):
    encoded = individual.embedding(model, device, max_depth, mapping)

    stats["called"] += + 1
    trials = 0
    mutated = None

    while mutated is None and trials < max_trials:
        stats["trials"] += 1
        trials += 1
        mask = (torch.randn_like(encoded) < ratio).float()
        noise = torch.randn_like(encoded) * mask * scaler
        decoded = model.decode(encoded + noise)

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
            stats["success"] += 1
        except (RuntimeError, IndexError):
            mutated = None

        if force_diffent and mutated == individual:
            mutated = None

    if mutated is None:
        # fallback to mutuniform if no valid mutation is found in 3 trials
        #mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]
        mutated = individual
        stats["fallbacked"] += 1

    if mutated == individual:
        stats["unchanged"] += 1

    return mutated,


def mut_rev_cosine_dist(individual, pset, mapping, max_depth, model, distance, stats, force_diffent=False,
                        max_trials=3):
    def gram_schmidt_orthogonalization(A, random_vector):
        # Project the random vector onto A
        projection = torch.dot(random_vector, A) / torch.dot(A, A) * A
        orthogonal_vector = random_vector - projection

        # Normalize the orthogonal vector to make it a unit vector
        orthogonal_unit_vector = orthogonal_vector / torch.norm(orthogonal_vector)
        return orthogonal_unit_vector

    def torch_cos_sim(v, cos_theta, n_vectors=1, EXACT=True):

        """
        EXACT - if True, all vectors will have exactly cos_theta similarity.
                if False, all vectors will have >= cos_theta similarity
        v - original vector (1D tensor)
        cos_theta -cos similarity in range [-1,1]
        """
        # unit vector in direction of v
        u = v / torch.norm(v)
        u = u.unsqueeze(0).repeat(n_vectors, 1)
        # random vector with elements in range [-1,1]
        r = torch.rand([n_vectors, len(v)]) * 2 - 1
        # unit vector perpendicular to v and u
        uperp = torch.stack([r[i] - (torch.dot(r[i], u[i]) * u[i]) for i in range(len(u))])
        uperp = uperp / (torch.norm(uperp, dim=1).unsqueeze(1).repeat(1, v.shape[0]))

        if not EXACT:
            cos_theta = torch.rand(n_vectors) * (1 - cos_theta) + cos_theta
            cos_theta = cos_theta.unsqueeze(1).repeat(1, v.shape[0])

            # w is the linear combination of u and uperp with coefficients costheta
        # and sin(theta) = sqrt(1 - costheta**2), respectively:
        w = cos_theta * u + torch.sqrt(1 - torch.tensor(cos_theta) ** 2) * uperp
        return w

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

    def generate_vector_with_cosine_distance(A, D):
        A_norm = F.normalize(A, p=2, dim=0)
        # Create a random vector orthogonal to A
        random_vector = torch.randn_like(A)
        orthogonal_vector = random_vector - torch.dot(random_vector, A_norm) * A_norm
        orthogonal_vector = F.normalize(orthogonal_vector, p=2, dim=0)
        # Calculate the cosine of the desired angle
        cos_theta = 1 - D
        # Generate the vector B using the combination of A_norm and orthogonal_vector
        sin_theta = torch.sqrt(torch.tensor(1 - cos_theta ** 2))  # Convert to tensor
        B = cos_theta * A_norm + sin_theta * orthogonal_vector
        # Ensure B is a unit vector
        #B = F.normalize(B, p=2, dim=0)

        return B


    stats["called"] += + 1
    encoded = individual.embedding(model, device, max_depth, mapping)
    trials = 0
    mutated = None

    while mutated is None and trials < max_trials:
        stats["trials"] += 1
        trials += 1
        original_shape = encoded.shape
        #distance = np.random.uniform(0, distance)
        #distanced = generate_vector_with_cosine_distance(encoded.flatten(), distance)
        distanced = torch_cos_sim(encoded.flatten(), distance, n_vectors=1, EXACT=True).squeeze(0)
        decoded = model.decode(distanced.reshape(original_shape))

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
            stats["success"] += 1
        except (RuntimeError, IndexError):
            mutated = None

        if force_diffent and mutated == individual:
            mutated = None

    if mutated is None:
        # fallback to mutuniform if no valid mutation is found in 3 trials
        #mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]
        mutated = individual
        stats["fallbacked"] += 1

    if mutated == individual:
        stats["unchanged"] += 1

    return mutated,


def mut_rev_euclid_dist(individual, pset, mapping, max_depth, model, distance, stats, force_diffent=False, max_trials=3):

    def generate_vector_with_distance(A, D):
        # Ensure A is a tensor
        A = torch.tensor(A, dtype=torch.float32)
        # Generate a random vector orthogonal to A
        random_vector = torch.randn_like(A)
        # Project the random vector onto the plane orthogonal to A
        projection = random_vector - torch.dot(random_vector, A) / torch.dot(A, A) * A
        # Normalize the projection to have a unit length
        unit_vector = projection / torch.norm(projection)
        # Scale the unit vector by the desired distance D
        scaled_vector = unit_vector * D
        # Add the scaled vector to A to get B
        B = A + scaled_vector
        return B

    stats["called"] += + 1
    encoded = individual.embedding(model, device, max_depth, mapping)
    trials = 0
    mutated = None

    while mutated is None and trials < max_trials:
        stats["trials"] += 1
        trials += 1
        original_shape = encoded.shape
        distanced = generate_vector_with_distance(encoded.flatten(), distance)
        decoded = model.decode(distanced.reshape(original_shape))

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
            stats["success"] += 1
        except (RuntimeError, IndexError):
            mutated = None

        if force_diffent and mutated == individual:
            mutated = None

    if mutated is None:
        # fallback to mutuniform if no valid mutation is found in 3 trials
        #mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]
        mutated = individual
        stats["fallbacked"] += 1

    if mutated == individual:
        stats["unchanged"] += 1

    return mutated,


def de_mut(individual, pset, mapping, max_depth, model, population, stats, F=0.5, CR=0.5, force_diffent=False, max_trials=3):
    mutated = None
    trials = 0
    stats["called"] += 1
    while mutated is None and trials < max_trials:
        stats["trials"] += 1
        trials += 1
        inds = random.sample(population, 3)
        x1, x2, x3 = [ind.embedding(model, device, max_depth, mapping) for ind in inds]

        mutant_vector = x1 + F * (x2 - x3)
        target = individual.embedding(model, device, max_depth, mapping)

        mask = (torch.randn_like(x1) < CR).float()
        mask[torch.randint(0, len(x1), (1,)).item()] = 1.0

        trial_vector = mutant_vector * mask + target * (1 - mask)
        decoded = model.decode(trial_vector, limited_dictionary=mapping)

        mutated = SymRegTree.from_tokenized_tree(decoded, mapping, pset)
        try:
            mutated.padded = mutated.add_padding(pset, max_depth)
            mutated.fitness = individual.fitness
            mutated.pset = individual.pset
            stats["success"] += 1
        except (RuntimeError, IndexError):
            mutated = None

        if force_diffent and mutated == individual:
            mutated = None

    if mutated is None:
        stats["fallbacked"] += 1
        #mutated = mutUniform(individual, expr=partial(gp.genFull, min_=0, max_=max_depth), pset=pset)[0]
        mutated = individual

    if mutated == individual:
        stats["unchanged"] += 1

    return mutated,
