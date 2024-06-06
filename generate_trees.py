import os
from pprint import pprint

import pmlb
import argparse

from gp.Generate_data import generate_random_trees, generate_trees_from_evolution
import numpy as np
from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import get_mapping


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    res = '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    res.replace(".", "_")
    return res


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generate trees")

    parser.add_argument("--n_random_trees", type=int, default=100000)
    parser.add_argument("--min_depth", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="4544_GeographicalOriginalofMusic")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--n_generations", type=int, default=55)
    parser.add_argument("--n_trees_per_generation", type=int, default=900)
    parser.add_argument("--skip_first_n_generations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="datasets")
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    stats = {}
    dataset = pmlb.fetch_data(args.dataset, local_cache_dir="datasets/pmlb_cache", dropna=True)
    pset = create_basic_symreg_pset(dataset)
    mapping = get_mapping(pset, ["PAD", "UNKNOWN"])

    stats["dictionary_size"] = len(mapping)
    print("Dictionary size: ", stats["dictionary_size"])

    print(f"=== Generating random trees with {args.dataset} ===")
    random_trees = generate_random_trees(pset, args.n_random_trees, min_depth=args.min_depth, max_depth=args.max_depth)
    stats["n_random_trees"] = len(random_trees)
    print(f"Generated {stats["n_random_trees"]} random trees")
    stats["random_trees_depth_avg"] = sum([tree.height for tree in random_trees]) / len(random_trees)
    print(f"Average depth of generated random trees: {stats["random_trees_depth_avg"]}")
    tokenized_trees = [tree.tokenize(pset, args.max_depth) for tree in random_trees]
    random_trees_array = np.array(tokenized_trees)
    stats["random_trees_pad_percentage"] = np.sum(random_trees_array == 0) / random_trees_array.size * 100
    print(f"Percentage of PAD tokens in generated random trees: {stats["random_trees_pad_percentage"]} %")

    print()
    print(f"=== Generating trees from evolution with {args.dataset} ===")
    evolution_trees = generate_trees_from_evolution(pset, dataset, n_generations=55, n_trees_per_generation=900,
                                                    min_depth=args.min_depth, max_depth=args.max_depth,
                                                    skip_first_n_generations=5)
    stats["n_from_evolution_trees"] = len(evolution_trees)
    print(f"Generated {stats["n_from_evolution_trees"]} trees from evolution")
    stats["from_evolution_trees_depth_avg"] = sum([tree.height for tree in evolution_trees]) / len(evolution_trees)
    print(f"Average depth of trees generated from evolution: {stats["from_evolution_trees_depth_avg"]}")
    tokenized_trees = [tree.tokenize(pset, args.max_depth) for tree in random_trees]
    evolution_trees_array = np.array(tokenized_trees)
    print(f"Percentage of PAD tokens in trees generated from evolution: "
          f"{np.sum(evolution_trees_array == 0) / evolution_trees_array.size * 100} %")

    print()
    trees = random_trees + evolution_trees
    print(f"Generated {len(trees)} trees in total")
    print("Average depth of all trees: ", sum([tree.height for tree in trees]) / len(trees))
    tokenized_trees = [tree.tokenize(pset, args.max_depth) for tree in trees]
    trees_array = np.array(tokenized_trees, dtype=int)
    np.random.shuffle(trees_array)
    print(f"Tree array shape: {trees_array.shape}")
    print(f"Percentage of PAD tokens in all trees: {np.sum(trees_array == 0) / trees_array.size * 100} %")

    if args.output_file is None:
        args.output_file = (f"{args.dataset}_depth{args.min_depth}-{args.max_depth}"
                            f"-{human_format(len(trees_array))}.txt")

    output_path = os.path.join(args.output_dir, args.output_file)
    print(f"Saving trees to {output_path}")
    np.savetxt(output_path, trees_array)
    with open(f"{output_path}.info", "w") as f:
        info = {"args": args, "mapping": mapping, "pset": pset}
        pprint(info, stream=f)


if __name__ == '__main__':
    main()
