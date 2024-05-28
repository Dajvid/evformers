import copy

from deap import gp


# TOKEN_MAPPING = {
#     "PAD": 0,
#     "UNKNOWN": 1,
#
# }


def hash_elements(elements, pset):
    mapping = get_mapping(pset, ["PAD", "UNKNOWN"])

    hashed = [mapping[element] for element in elements]

    return hashed


def get_mapping(pset, extra_elements=None):
    if extra_elements is None:
        extra_elements = []

    mapping = {}
    elems = extra_elements + list(pset.mapping.keys())
    for i, element in enumerate(elems):
        mapping[element] = i

    return mapping


class SymRegTree(gp.PrimitiveTree):

    def tokenize(self, pset, max_depth):
        padded = self.add_padding(max_depth, max_depth)
        return hash_elements(padded, pset)

    def add_padding(self, pset, max_depth):
        padded = [x.name for x in self]
        stack = [0]
        current_depth = 0
        insertions = []
        for i in range(len(self)):
            if self[i].arity > 2:
                raise ValueError("Arity of function must be less than or equal to 2")
            if self[i].arity < 2:
                insertions.append((i, (2 - self[i].arity) * (2 ** (max_depth - current_depth) - 1)))
            if self[i].arity == 0:
                current_depth = stack.pop()
            else:
                current_depth += 1
                stack.extend([current_depth] * self[i].arity)
        for i, n in reversed(insertions):
            padded[i+1:i+1] = ["PAD"] * int(n)

        return padded
