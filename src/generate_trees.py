import os
from pprint import pprint

import pmlb
import argparse

from gp.Generate_data import generate_random_trees, generate_trees_from_evolution, even_parity_truth_table
import numpy as np
from gp.Pset import create_basic_symreg_pset, create_basic_logic_pset
from gp.sym_reg_tree import get_mapping
import pickle

from gp.Fitness import eval_symb_reg_pmlb, binary_regression_fitness


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
    parser.add_argument("--dataset", type=str, default="505_tecator")
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--n_generations", type=int, default=55)
    parser.add_argument("--n_trees_per_generation", type=int, default=900)
    parser.add_argument("--skip_first_n_generations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="../datasets")
    parser.add_argument("--SOT", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--EOT", type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    args = parse_args(argv)
    stats = {}
    fitness = eval_symb_reg_pmlb
    if args.dataset.split('-')[1] == "parity":
        try:
            dataset = even_parity_truth_table(int(args.dataset.split('-')[0]))
            pset = create_basic_logic_pset(dataset)
            fitness = binary_regression_fitness
        except IndexError:
            pass
    else:
        dataset = pmlb.fetch_data(args.dataset, local_cache_dir=os.path.join(args.output_dir, "pmlb_cache"), dropna=True)
        pset = create_basic_symreg_pset(dataset)
    mapping = get_mapping(pset, ["PAD", "UNKNOWN"])
    if args.SOT:
        mapping_SOT = get_mapping(pset, ["PAD", "UNKNOWN", "SOT"])
    if args.SOT and args.EOT:
        mapping_EOT = get_mapping(pset, ["PAD", "UNKNOWN", "SOT", "EOT"])

    stats["dictionary_size"] = len(mapping)
    print("Dictionary size: ", stats["dictionary_size"])

    print(f"=== Generating random trees with {args.dataset} ===")
    random_trees = generate_random_trees(pset, args.n_random_trees, min_depth=args.min_depth, max_depth=args.max_depth)
    [tree.simplify() for tree in random_trees]

    stats["n_random_trees"] = len(random_trees)
    print(f"Generated {stats["n_random_trees"]} random trees")
    stats["random_trees_depth_avg"] = sum([tree.height for tree in random_trees]) / len(random_trees)
    print(f"Average depth of generated random trees: {stats["random_trees_depth_avg"]}")
    tokenized_trees = [tree.tokenize(args.max_depth, mapping) for tree in random_trees]
    random_trees_array = np.array(tokenized_trees)
    stats["random_trees_pad_percentage"] = np.sum(random_trees_array == 0) / random_trees_array.size * 100
    print(f"Percentage of PAD tokens in generated random trees: {stats["random_trees_pad_percentage"]} %")

    print()
    print(f"=== Generating trees from evolution with {args.dataset} ===")
    if not args.n_generations == 0:
        evolution_trees = generate_trees_from_evolution(pset, dataset, n_generations=args.n_generations, fitness=fitness,
                                                        n_trees_per_generation=args.n_trees_per_generation,
                                                        min_depth=args.min_depth, max_depth=args.max_depth,
                                                        skip_first_n_generations=5)
        stats["n_from_evolution_trees"] = len(evolution_trees)
        print(f"Generated {stats["n_from_evolution_trees"]} trees from evolution")
        stats["from_evolution_trees_depth_avg"] = sum([tree.height for tree in evolution_trees]) / len(evolution_trees)
        print(f"Average depth of trees generated from evolution: {stats["from_evolution_trees_depth_avg"]}")
        tokenized_trees = [tree.tokenize(args.max_depth, mapping) for tree in evolution_trees]
        evolution_trees_array = np.array(tokenized_trees)
        stats["from_evolution_trees_pad_percentage"] = np.sum(evolution_trees_array == 0) / evolution_trees_array.size * 100
        print(f"Percentage of PAD tokens in trees generated from evolution: "
              f"{stats["from_evolution_trees_pad_percentage"]} %")
    else:
        evolution_trees = []

    print()
    trees = random_trees + evolution_trees
    print(f"Generated {len(trees)} trees in total")
    print("Average depth of all trees: ", sum([tree.height for tree in trees]) / len(trees))
    tokenized_trees = [tree.tokenize(args.max_depth, mapping) for tree in trees]
    trees_array = np.array(tokenized_trees, dtype=int)
    randomize = np.arange(len(trees_array))
    np.random.shuffle(randomize)
    trees_array = trees_array[randomize]
    print(f"Tree array shape: {trees_array.shape}")
    print(f"Percentage of PAD tokens in all trees: {np.sum(trees_array == 0) / trees_array.size * 100} %")

    if args.SOT:
        tokenized_trees_SOT = [tree.tokenize(args.max_depth, mapping_SOT, add_SOT=True) for tree in trees]
        trees_SOT_array = np.array(tokenized_trees_SOT, dtype=int)
        trees_SOT_array = trees_SOT_array[randomize]
        if fitness == binary_regression_fitness:
            tokenized_targets_SOT = [tree.simplify().tokenize(args.max_depth, mapping_SOT, add_SOT=True) for tree in trees]
            tree_targets_SOT = np.array(tokenized_targets_SOT, dtype=int)
            tree_targets_SOT = tree_targets_SOT[randomize]
    if args.EOT:
        tokenized_trees_SOT = [tree.tokenize(args.max_depth, mapping_EOT, add_SOT=True,
                                             add_EOT=True) for tree in trees]
        trees_EOT_array = np.array(tokenized_trees_SOT, dtype=int)
        trees_EOT_array = trees_EOT_array[randomize]

    if args.output_name is None:
        args.output_name = f"{args.dataset}-depth-{args.min_depth}-{args.max_depth}-{human_format(len(trees_array))}"

    output_dir_path = os.path.join(args.output_dir, args.output_name)

    output_name_path = f"{os.path.join(output_dir_path, args.output_name)}"
    os.makedirs(output_dir_path, exist_ok=True)

    print(f"Saving trees to {output_name_path}.data")
    np.savetxt(f"{output_name_path}.data", trees_array)
    with open(f"{output_name_path}.info", "w") as f:
        info = {"args": args, "mapping": mapping, "pset": pset}
        pprint(info, stream=f)
    with open(f"{output_name_path}.dict", "wb") as f:
        pickle.dump(mapping, f)
    if args.SOT:
        np.savetxt(f"{output_name_path}-SOT.data", trees_SOT_array)
        if fitness == binary_regression_fitness:
            np.savetxt(f"{output_name_path}-SOT.targets", tree_targets_SOT)
        with open(f"{output_name_path}-SOT.info", "w") as f:
            info = {"args": args, "mapping": mapping_SOT, "pset": pset}
            pprint(info, stream=f)
        with open(f"{output_name_path}-SOT.dict", "wb") as f:
            pickle.dump(mapping_SOT, f)
    if args.EOT:
        np.savetxt(f"{output_name_path}-EOT.data", trees_EOT_array)
        with open(f"{output_name_path}-EOT.info", "w") as f:
            info = {"args": args, "mapping": mapping_EOT, "pset": pset}
            pprint(info, stream=f)
        with open(f"{output_name_path}-EOT.dict", "wb") as f:
            pickle.dump(mapping_SOT, f)


if __name__ == '__main__':
    main()
