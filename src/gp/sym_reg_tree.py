from deap import gp


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
        def hash_elements(elements):
            mapping = get_mapping(pset, ["PAD", "UNKNOWN"])
            hashed = [mapping[element] for element in elements]
            return hashed

        padded = self.add_padding(max_depth, max_depth)
        return hash_elements(padded)

    def add_padding(self, pset, max_depth):
        if len(self) > 2 ** (max_depth + 1) - 1:
            raise ValueError("Tree is too deep")
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
                stack.extend([current_depth] * (self[i].arity - 1))
        if len(self) + sum([n for _, n in insertions]) != 2 ** (max_depth + 1) - 1:
            raise RuntimeError("Tree is not full")
        for i, n in reversed(insertions):
            if n != 0:
                padded[i+1:i+1] = ["PAD"] * int(n)

        return padded

    def compile(self):
        return gp.compile(self, self.pset)
