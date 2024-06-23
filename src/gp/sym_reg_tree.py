import re
from collections import namedtuple
from typing import List, Dict
from sympy import And, Or, Not

import sympy
import torch
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

    def tokenize(self, max_depth, mapping, add_SOT=False, add_EOT=False):
        def hash_elements(elements):
            hashed = [mapping[element] for element in elements]
            return hashed

        def get_last_non_pad_index(padded):
            non_pad_i = 1
            while non_pad_i < len(padded):
                if padded[-non_pad_i] != "PAD":
                    break
                non_pad_i += 1
            return len(padded) - non_pad_i

        padded = self.add_padding(max_depth, max_depth)
        if add_SOT:
            padded = ["SOT"] + padded
        if add_EOT:
            last_non_pad_index = get_last_non_pad_index(padded)
            padded = padded[:last_non_pad_index + 1] + ["EOT"] + padded[last_non_pad_index + 1:]
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

    def embedding(self, model, device, max_depth, mapping):
        if not hasattr(self, "emb"):
            self.emb = model.encode(torch.tensor(self.tokenize(max_depth, mapping,
                                                               add_SOT="SOT" in mapping.keys()), device=device))

        return self.emb

    def decoder_embedding(self, model, device, max_depth, mapping):
        return model.decoder_embedding(torch.tensor(self.tokenize(max_depth, mapping,
                                                                  add_SOT="SOT" in mapping.keys()), device=device))

    def compile(self):
        return gp.compile(self, self.pset)

    def simplify(self):
        def binary_and(*args):
            if len(args) == 2:
                return f"And({args[0]}, {args[1]})"
            else:
                return f"And({args[0]}, {binary_and(*args[1:])})"

        def binary_or(*args):
            if len(args) == 2:
                return f"Or({args[0]}, {args[1]})"
            else:
                return f"Or({args[0]}, {binary_or(*args[1:])})"

        def to_binary_ops(expr):
            if isinstance(expr, And):
                args = list(map(to_binary_ops, expr.args))
                return binary_and(*args)
            elif isinstance(expr, Or):
                args = list(map(to_binary_ops, expr.args))
                return binary_or(*args)
            elif isinstance(expr, Not):
                return f"Not({to_binary_ops(expr.args[0])})"
            else:
                return str(expr)

        def dfs(expr):
            """Perform a DFS traversal on the expression tree and collect operators."""
            if expr.is_Atom:
                return [expr]
            traversal = []

            traversal.append(expr.func)
            for arg in expr.args:
                traversal.extend(dfs(arg))

            return traversal

        simplified_str = to_binary_ops(sympy.simplify(sympy.sympify(str(self))))
        #simplified_tree_dfs = dfs(simplified)
        #simplified_str = re.sub(r"Symbol\('(.*?)'\)", lambda x: x.group(1), str(sympy.srepr(simplified)))
        simplified_str = simplified_str.replace("true", "True").replace("false", "False")
        #simplified_str = simplified_str.replace("(", " ").replace(")", " ").replace(",", "")
        elements = simplified_str.split()
        #instanciated = [self.pset.mapping[x.replace("true", "True").replace("false", "False")] for x in simplified_tree_dfs]
        simplified_tree = SymRegTree.from_string(simplified_str, self.pset)
       # simplified_tree = SymRegTree(instanciated)
        simplified_tree.pset = self.pset
        return simplified_tree

    @classmethod
    def from_tokenized_tree(cls, tokenized: List, mapping: Dict, pset: gp.PrimitiveSet):
        inv_mapping = {v: k for k, v in mapping.items()}
        demaped = [inv_mapping[i] for i in tokenized]
        depaded = [x for x in demaped if x not in ["PAD", "SOT", "EOT"]]
        instanciated = [pset.mapping[x] for x in depaded]
        tree = cls(instanciated)
        tree.pset = pset
        # TODO include fitness to make it a full individual compatible with deap

        return cls(instanciated)

    @classmethod
    def from_string(cls, string, pset):
        tree = super().from_string(string, pset)
        tree.pset = pset
        return tree
