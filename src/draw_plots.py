import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmlb
import seaborn as sns
from pathlib import Path

import torch

from gp.Pset import create_basic_symreg_pset, create_basic_logic_pset
from gp.sym_reg_tree import SymRegTree, get_mapping
from gpformer.model import Transformer
from gp.Generate_data import even_parity_truth_table
from gp.Fitness import binary_regression_fitness

figsize = (14, 8)
sns.set(font_scale=1)

def add_property(folder: str, property: str, value: float):
    pathlist = list(Path(folder).rglob('*.pkl'))
    for i, path in enumerate(pathlist):
        print(f"Altering {i + 1} / {len(pathlist)}: {path}")
        df = pd.read_pickle(path)
        df[property] = value
        df.to_pickle(path)


def rename_property(folder: str, old_property: str, new_property: str):
    pathlist = list(Path(folder).rglob('*.pkl'))
    for i, path in enumerate(pathlist):
        print(f"Altering {i + 1} / {len(pathlist)}: {path}")
        df = pd.read_pickle(path)
        df[new_property] = df[old_property]
        df.drop(columns=[old_property], inplace=True)
        df.to_pickle(path)


def load_data(folder: str = "../runs/evolution/--p-cross"):
    pathlist = list(Path(folder).rglob('*.pkl'))
    dataframes = []
    for i, path in enumerate(pathlist):
        print(f"Loading {i + 1} / {len(pathlist)}: {path} \r", end="")
        dataframes.append(pd.read_pickle(path))

    return pd.concat(dataframes)


def plot_data(df: pd.DataFrame, feature: str, x_label: str, path: str = None):
    df = df.copy()
    df2 = df.copy()
    plt.figure()
    df.rename(columns={"fit-min": "fitness", feature: x_label}, inplace=True)
    grouped = df.groupby([x_label, "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x=x_label)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_succes_rate(df: pd.DataFrame, feature: str, x_label: str, path: str = None):
    df = df.copy()
    plt.figure()
    df.rename(columns={feature: x_label}, inplace=True)
    grouped = df.groupby([x_label]).agg({"mut_success": "sum", "mut_called": "sum"})
    grouped["Mutation success rate"] = grouped["mut_success"] / grouped["mut_called"]
    sns.lineplot(grouped.reset_index(), y="Mutation success rate", x=x_label)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_convergence(data, xlabel, ylabel, out_path, fit_evaluations=10000, logscale=False, visualize=False):
    """
    Plot convergence of data and save the plot to a file.

    :param data (dict): A dictionary containing the data to be plotted. The keys represent labels, and the values are
            numpy arrays of data points.
    :param xlabel: The label for the x-axis.
    :parame ylabel: The label for the y-axis.
    :param out_path: The file path to save the plot.
    :param logscale: If True, the x-axis is plotted on a logarithmic scale. Defaults to False.
    :param visualize: If True, display the plot interactively. Defaults to False.
    """
    plt.figure(figsize=figsize)
    plt.plot()
    for key, values in data.items():
        #grouped = values.groupby(["gen", "run_id"]).agg({"fit-min": ["min", "avg", "max"]})
        values = values.groupby(["gen"]).agg({"fit-min": ["min", "mean", "max"]}).reset_index()
        x = np.linspace(1, fit_evaluations, num=len(values), dtype=np.int64)
        if logscale:
            plt.xscale('log')
        plt.plot(x, values["fit-min", "mean"], label=key, scalex=False)
        plt.fill_between(x, values["fit-min", "min"], values["fit-min", "max"], alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    if visualize:
        plt.show()
    plt.close()


def hamming_distance_heatmap(number_of_inputs, trees=None):
    pset = create_basic_logic_pset(even_parity_truth_table(3))
    trees = {
        "T1": SymRegTree.from_string("And(ARG0, True)", pset=pset),
        "T1R": SymRegTree.from_string("And(True, ARG0)", pset=pset),
        "T1EQ": SymRegTree.from_string("And(Not(False), ARG0)", pset=pset),
        "T1S": SymRegTree.from_string("ARG0", pset=pset),

        "T2": SymRegTree.from_string("Or(ARG0, ARG2)", pset=pset),
        "T2R": SymRegTree.from_string("Or(ARG2, ARG0)", pset=pset),
        "T2EQ": SymRegTree.from_string("Or(Not(Not(ARG2)), Not(Not(ARG0)))", pset=pset),

        "T3": SymRegTree.from_string("And(Or(ARG0, ARG1), True)", pset=pset),
        "T3R": SymRegTree.from_string("And(True, Or(ARG0, ARG1))", pset=pset),
        "T3EQ": SymRegTree.from_string("And(Or(ARG0, ARG1), Or(ARG1, Not(False)))", pset=pset),
        "T3S": SymRegTree.from_string("Or(ARG0, ARG1)", pset=pset),
    }

    inputs = pd.DataFrame(list(itertools.product([0, 1], repeat=number_of_inputs)),
                          columns=[f'X{i}' for i in range(number_of_inputs)])
    matrix = np.zeros((len(trees.items()), len(trees)))
    for i, (key1, tree1) in enumerate(trees.items()):
        func = tree1.compile()
        tree1_outputs = inputs.apply(lambda x: func(*x), axis=1)
        for j, (key2, tree2) in enumerate(trees.items()):
            matrix[i, j] = binary_regression_fitness(tree2, inputs, tree1_outputs)[0]

    sns.heatmap(matrix)
    plt.show()


def similarity_heatmap(model_weights="../model-tecator-0-6-SOT.pth", dataset="505_tecator", SOT=True, masked=False,
                       measure="cos_dist", ignore_pad=False, aggregate=None, use_decoder_embedding=False,
                       kwargs=None):
    if dataset in pmlb.dataset_names:
        pset = create_basic_symreg_pset(pmlb.fetch_data(dataset))
        trees = {
            "T1": SymRegTree.from_string("add(ARG0, 1.0)", pset=pset),
            "T1R": SymRegTree.from_string("add(1.0, ARG0)", pset=pset),
            "T1EQ": SymRegTree.from_string("add(ARG0, add(0.5, 0.5))", pset=pset),

            "T2": SymRegTree.from_string("div(ARG0, 0.5)", pset=pset),
            "T2R": SymRegTree.from_string("div(0.5, ARG0)", pset=pset),
            "T2EQ": SymRegTree.from_string("mul(ARG0, add(1.0, 1.0))", pset=pset),

            "T3": SymRegTree.from_string("mul(ARG0, 0.5)", pset=pset),
            "T3R": SymRegTree.from_string("mul(0.5, ARG0)", pset=pset),
            "T3EQ": SymRegTree.from_string("div(ARG0, add(1.0, 1.0))", pset=pset),

            "T4": SymRegTree.from_string("add(div(ARG0, 0.5), 1.0)", pset=pset),
            "T4R": SymRegTree.from_string("add(1.0, div(ARG0, 0.5))", pset=pset),

            "T5": SymRegTree.from_string("div(ARG0, div(0.20000000000000018, ARG0))", pset=pset),
            "T5R": SymRegTree.from_string("div(div(0.20000000000000018, ARG0), ARG0)", pset=pset),

            "T6": SymRegTree.from_string("add(mul(ARG0, div(ARG0, 0.5)), 1.0)", pset=pset),
            "T6R": SymRegTree.from_string("add(1.0, mul(ARG0, div(ARG0, 0.5)))", pset=pset),

            "T7": SymRegTree.from_string("sub(mul(ARG0, 0.30000000000000004), sub(ARG0, 0.8))", pset=pset),
            "T7R": SymRegTree.from_string("sub(sub(ARG0, 0.8), mul(ARG0, 0.30000000000000004))", pset=pset),
        }
    else:
        pset = create_basic_logic_pset(even_parity_truth_table(3))
        trees = {
            "T1": SymRegTree.from_string("And(ARG0, True)", pset=pset),
            "T1R": SymRegTree.from_string("And(True, ARG0)", pset=pset),
            "T1EQ": SymRegTree.from_string("And(Not(False), ARG0)", pset=pset),
            "T1S": SymRegTree.from_string("ARG0", pset=pset),

            "T2": SymRegTree.from_string("Or(ARG0, ARG2)", pset=pset),
            "T2R": SymRegTree.from_string("Or(ARG2, ARG0)", pset=pset),
            "T2EQ": SymRegTree.from_string("Or(Not(Not(ARG2)), Not(Not(ARG0)))", pset=pset),

            "T3": SymRegTree.from_string("And(Or(ARG0, ARG1), True)", pset=pset),
            "T3R": SymRegTree.from_string("And(True, Or(ARG0, ARG1))", pset=pset),
            "T3EQ": SymRegTree.from_string("And(Or(ARG0, ARG1), Or(ARG1, Not(False)))", pset=pset),
            "T3S": SymRegTree.from_string("Or(ARG0, ARG1)", pset=pset),
        }

    extra_elements = ["PAD", "UNKNOWN"]
    if SOT:
        extra_elements.append("SOT")
    mapping = get_mapping(pset, extra_elements) if SOT else get_mapping(pset, ["PAD", "UNKNOWN"])
    if masked:
        max_token_id = max(list(mapping.values()))
        mapping["MASK"] = max_token_id + 1

    model = Transformer(mapping, 2 * 6, 2, 1, 1,
                        6, 2, ignore_pad=ignore_pad).to("cpu")
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))

    encoded_trees = {key: tree.embedding(model, "cpu", 6, mapping) for key, tree in trees.items()}
    decoded_trees = {key: str(SymRegTree.from_tokenized_tree(model.decode(tree), mapping, pset)) for key, tree in encoded_trees.items()}
    if use_decoder_embedding:
        trees = {key: tree.decoder_embedding(model, "cpu", 6, mapping) for key, tree in trees.items()}
    else:
        trees = encoded_trees
    matrix = np.zeros((len(trees), len(trees)))
    for i, (key1, tree1) in enumerate(trees.items()):
        for j, (key2, tree2) in enumerate(trees.items()):
            if i == j:
                matrix[i, j] = 0
            else:
                if measure == "cos_dist":
                    if aggregate == "sum":
                        matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(tree1.sum(dim=1), tree2.sum(dim=1), dim=0)
                    elif aggregate == "mean":
                        matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(tree1.mean(dim=0), tree2.mean(dim=0), dim=0)
                    else:
                        matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(tree1.flatten(), tree2.flatten(), dim=0)
                elif measure == "mse":
                    matrix[i, j] = torch.nn.functional.mse_loss(tree1, tree2)
                elif measure == "euclidean":
                    matrix[i, j] = torch.dist(tree1.flatten(), tree2.flatten(), p=2)

    sns.heatmap(matrix, xticklabels=trees.keys(), yticklabels=trees.keys(), **kwargs)
    # print({key: str(SymRegTree.from_tokenized_tree(model.decode(tree), mapping, pset)) for key, tree in trees.items()})
    print(decoded_trees)



def similarity_heatmaps():
    models=["../experiments/model_training/SOT_influence/SOT/elusage/2024-06-10_01-24-10_dhcpg197.fit.vutbr.cz/model.pth",
            "../experiments/model_training/SOT_influence/SOT/505_tecator/2024-06-10_05-18-18_dhcpf118.fit.vutbr.cz/model.pth",
            "../experiments/model_training/SOT_influence/SOT/fri_c4/2024-06-09_22-10-54_dhcpg197.fit.vutbr.cz/model.pth",
            "../experiments/model_training/SOT_influence/SOT/wind/2024-06-10_04-38-43_dhcpg197.fit.vutbr.cz/model.pth"]
    datasets = ["228_elusage", "505_tecator", "588_fri_c4_1000_100", "503_wind"]
    for model, dataset in zip(models, datasets):
        similarity_heatmap(model, dataset, ignore_pad=False, SOT=True)

    models = ["../experiments/model_training/pad_handling/ignore_pad/elusage/2024-06-09_13-21-27_dhcpg197.fit.vutbr.cz/model.pth",
              "../experiments/model_training/pad_handling/ignore_pad/505_tecator/2024-06-09_16-32-31_dhcpf118.fit.vutbr.cz/model.pth",
              "../experiments/model_training/pad_handling/ignore_pad/fri_c4/2024-06-09_20-34-26_dhcpf118.fit.vutbr.cz/model.pth",
              "../experiments/model_training/pad_handling/ignore_pad/wind/2024-06-09_14-55-20_dhcpf118.fit.vutbr.cz/model.pth"]
    datasets = ["228_elusage", "505_tecator", "588_fri_c4_1000_100", "503_wind"]
    for model, dataset in zip(models, datasets):
        similarity_heatmap(model, dataset, ignore_pad=True, SOT=False)


def benchmark_de_mut_cxOnePoint():
    df = load_data("../runs/evolution/de_mut_cxOnePoint/--mut-param")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "mut_param", "F", None)
    plot_succes_rate(df, "mut_param", "F", None)

    df = load_data("../runs/evolution/de_mut_cxOnePoint/--mut-ratio-param")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "mut_ratio_param", "CR", None)

    df = load_data("../runs/evolution/de_mut_cxOnePoint/--pop-size")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "pop_size", "Population size", None)

    df = load_data("../runs/evolution/de_mut_cxOnePoint/--p-mut")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "p_mut", "Mutation probability", "../plots/de_mut_cxOnePoint_p_mut.pdf")

    df = load_data("../runs/evolution/de_mut_cxOnePoint/--p-cross")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "p_cross", "Crossover probability", None)

    df = load_data("../runs/evolution/de_mut_cxOnePoint/--tournament-size")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "tournament_size", "Tournament size", None)


def plot_rev_euclid_dist_stats():
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    df = pd.read_pickle("../mut_rev_euclid_dist_no_force_diff_trials_3.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"] * 100
    df.rename(columns={"distance": "Mutation distance"}, inplace=True)
    sns.lineplot(data=df, x="Mutation distance", y="Changed after mutation [%]", ax=axes, label="Force change: False, trials: 3")

    df = pd.read_pickle("../mut_rev_euclid_dist_force_diff_trials_3.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"] * 100
    df.rename(columns={"distance": "Mutation distance"}, inplace=True)
    sns.lineplot(data=df, x="Mutation distance", y="Changed after mutation [%]", ax=axes, label="Force change: True, trials: 3")

    df = pd.read_pickle("../mut_rev_euclid_dist_force_diff_trials_10.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"] * 100
    df.rename(columns={"distance": "Mutation distance"}, inplace=True)
    sns.lineplot(data=df, x="Mutation distance", y="Changed after mutation [%]", ax=axes, label="Force change: True, trials: 10")

    df = pd.read_pickle("../mut_rev_euclid_dist_force_diff_trials_20.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"] * 100
    df.rename(columns={"distance": "Mutation distance"}, inplace=True)
    sns.lineplot(data=df, x="Mutation distance", y="Changed after mutation [%]", ax=axes, label="Force change: True, trials: 20")

    plt.savefig("../plots/mut_rev_euclid_dist_cxOnePoint_stats.pdf", bbox_inches='tight')


def plot_random_noise_stats():
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex=True)

    vmin = 0.0
    vmax = 1.0

    df = pd.read_pickle("../mut_add_random_noise_gaussian_stats_no_force_diff_trials_3.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"]
    df.rename(columns={"scaler": "Scaler", "ratio": "Ratio"}, inplace=True)
    sns.heatmap(df.pivot(index="Scaler", columns="Ratio", values="Changed after mutation [%]"), ax=axes[0, 0],
                vmin=vmin, vmax=vmax, cbar=False)
    axes[0, 0].set_title("Force change: False, trials: 3")

    df = pd.read_pickle("../mut_add_random_noise_gaussian_stats_force_diff_trials_3.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"]
    df.rename(columns={"scaler": "Scaler", "ratio": "Ratio"}, inplace=True)
    sns.heatmap(df.pivot(index="Scaler", columns="Ratio", values="Changed after mutation [%]"), ax=axes[0, 1],
                vmin=vmin, vmax=vmax, cbar=False)
    axes[0, 1].set_title("Force change: True, trials: 3")

    df = pd.read_pickle("../mut_add_random_noise_gaussian_stats_force_diff_trials_10.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"]
    df.rename(columns={"scaler": "Scaler", "ratio": "Ratio"}, inplace=True)
    sns.heatmap(df.pivot(index="Scaler", columns="Ratio", values="Changed after mutation [%]"), ax=axes[1, 0],
                vmin=vmin, vmax=vmax, cbar=False)
    axes[1, 0].set_title("Force change: True, trials: 10")

    df = pd.read_pickle("../mut_add_random_noise_gaussian_stats_force_diff_trials_20.pkl")
    df["Changed after mutation [%]"] = (250 - df["unchanged"]) / df["called"]
    df.rename(columns={"scaler": "Scaler", "ratio": "Ratio"}, inplace=True)
    sns.heatmap(df.pivot(index="Scaler", columns="Ratio", values="Changed after mutation [%]"), ax=axes[1, 1],
                vmin=vmin, vmax=vmax, cbar=False)
    axes[1, 1].set_title("Force change: True, trials: 20")

    cb = fig.colorbar(axes[0, 0].collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cb.set_label("Changed after mutation [%]")

    plt.savefig("../plots/mut_add_random_noise_gaussian_cxOnePoint_stats.pdf", bbox_inches='tight')


def plot_baseline():
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--pop-size")
    df.rename(columns={"fit-min": "fitness", "pop_size": "Population size"}, inplace=True)
    grouped = df.groupby(["Population size", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Population size", ax=axes[0])

    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--p-mut")
    df.rename(columns={"fit-min": "fitness", "p_mut": "Mutation probablity"}, inplace=True)
    grouped = df.groupby(["Mutation probablity", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Mutation probablity", ax=axes[1])

    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--tournament-size")
    df.rename(columns={"fit-min": "fitness", "tournament_size": "Tournament size"}, inplace=True)
    grouped = df.groupby(["Tournament size", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Tournament size", ax=axes[2])

    plt.savefig("../plots/baseline_tuning.pdf", bbox_inches='tight')

    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--tournament-size/505_tecator/--tournament-size-10/")
    plot_convergence({"Baseline": df}, "Fitness evaluations", "Fitness",
                     "../plots/baseline_convergence.pdf", fit_evaluations=5000, logscale=False,
                     visualize=True)


def plot_euclid_dist():
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--mut-param")
    df.rename(columns={"fit-min": "fitness", "mut_param": "Mutation distance"}, inplace=True)
    df["Mutation distance"] = df["Mutation distance"].astype(int)
    grouped = df.groupby(["Mutation distance", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Mutation distance", ax=axes[0])

    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--pop-size")
    df.rename(columns={"fit-min": "fitness", "pop_size": "Population size"}, inplace=True)
    grouped = df.groupby(["Population size", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Population size", ax=axes[1])

    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--p-mut")
    df.rename(columns={"fit-min": "fitness", "p_mut": "Mutation probability"}, inplace=True)
    grouped = df.groupby(["Mutation probability", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Mutation probability", ax=axes[2])

    plt.savefig("../plots/mut_rev_euclid_dist_cxOnePoint_tuning.pdf", bbox_inches='tight')

    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--p-mut/505_tecator/--p-mut-0.9/")
    plot_convergence({"MutRevEuclidDist": df}, "Fitness evaluations", "Fitness",
                     "../plots/mut_rev_euclid_dist_cxOnePoint_convergence.pdf", fit_evaluations=5000, logscale=False,
                     visualize=True)


def plot_comparison():
    baseline_df = load_data("../runs/evolution/mutUniform_cxOnePoint/--tournament-size/505_tecator/--tournament-size-10/")
    euclid_dist_df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--p-mut/505_tecator/--p-mut-0.9/")
    random_noise_df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--p-mut/505_tecator/--p-mut-0.9/")
    plot_convergence({
        "Baseline": baseline_df,
        "MutAddRandomNoise": random_noise_df,
        "MutReverseEuclidDistance": euclid_dist_df
    }, "Fitness evaluations", "Fitness",
                     "../plots/comparison_convergence.pdf",
                     fit_evaluations=5000, logscale=False,
                     visualize=True)

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    baseline_df["variant"] = "Baseline"
    euclid_dist_df["variant"] = "MutReverseEuclidDistance"
    random_noise_df["variant"] = "MutAddRandomNoise"
    df = pd.concat([baseline_df, euclid_dist_df, random_noise_df])
    df.rename(columns={"fit-min": "Fitness"}, inplace=True)
    sns.boxplot(data=df, y="Fitness", x="variant", ax=axes)
    plt.savefig("../plots/comparison_boxplot.pdf", bbox_inches='tight')


def plot_random_noise():
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-param")
    df.rename(columns={"fit-min": "fitness", "mut_param": "Scaler"}, inplace=True)
    grouped = df.groupby(["Scaler", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Scaler", ax=axes[0, 0])

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-ratio-param")
    df.rename(columns={"fit-min": "fitness", "mut_ratio_param": "Ratio"}, inplace=True)
    grouped = df.groupby(["Ratio", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Ratio", ax=axes[0, 1])

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--p-mut")
    df.rename(columns={"fit-min": "fitness", "p_mut": "Mutation probability"}, inplace=True)
    grouped = df.groupby(["Mutation probability", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Mutation probability", ax=axes[1, 0])

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--pop-size")
    df.rename(columns={"fit-min": "fitness", "pop_size": "Population size"}, inplace=True)
    grouped = df.groupby(["Population size", "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x="Population size", ax=axes[1, 1])

    plt.savefig("../plots/mut_add_random_noise_gaussian_cxOnePoint_tuning.pdf", bbox_inches='tight')

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--p-mut/505_tecator/--p-mut-0.9/")
    plot_convergence({"MutReverseEuclidDistance": df}, "Fitness evaluations", "Fitness",
                     "../plots/mut_add_random_noise_gaussian_cxOnePoint_convergence.pdf",
                     fit_evaluations=5000, logscale=False,
                     visualize=True)

def benchmark_mutUniform_cxOnePoint():
    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--pop-size")
    plot_data(df, "pop_size", "Population size", None)

    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--p-mut")
    plot_data(df, "p_mut", "Mutation probability", None)

    df = load_data("../runs/evolution/mutUniform_cxOnePoint/--tournament-size")
    plot_data(df, "tournament_size", "Tournament size", None)


def benchmark_rev_cosine_dist_cxOnePoint():
    df = load_data("../runs/evolution/mut_rev_cosine_dist_cxOnePoint/--mut-param")
    plot_data(df, "mut_param", "Cosine distance", None)
    plot_succes_rate(df, "mut_param", "Mutation succes rate", None)


def benchmark_rev_euclid_dist_cxOnePoint():
    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--mut-param")
    plot_data(df, "mut_param", "Euclidean distance", None)
    plot_succes_rate(df, "mut_param", "Mutation succes rate", None)

    df = load_data("../runs/evolution/mut_rev_euclid_dist_cxOnePoint/--pop-size")
    plot_data(df, "pop_size", "Population size", None)
    plot_succes_rate(df, "pop_size", "Mutation succes rate", None)

def benchmark_random_noise_cxOnePoint():
    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-param")
    plot_data(df, "mut_param", "Scaler", None)
    plot_succes_rate(df, "mut_param", "Mutation succes rate", None)

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-ratio-param")
    plot_data(df, "mut_ratio_param", "Mutation ratio", None)
    plot_succes_rate(df, "mut_ratio_param", "Mutation succes rate", None)

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--pop-size")
    plot_data(df, "pop_size", "Population size", None)
    plot_succes_rate(df, "pop_size", "Mutation succes rate", None)

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--p-mut")
    plot_data(df, "p_mut", "Mutation probability", None)
    plot_succes_rate(df, "p_mut", "Mutation probability", None)


def model_plots():
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)

    dropout_influence_df = load_data("../experiments/model_training/dropout_influence/").reset_index()
    last_epochs = dropout_influence_df[dropout_influence_df["epoch"] == 49]
    last_epochs = last_epochs.rename(columns={"val_sequence_accuracy": "Validation sequence accuracy",
                                              "dropout": "Dropout"}, inplace=False)
    sns.boxplot(data=last_epochs, x="Dropout", y="Validation sequence accuracy", ax=axes[0, 0])

    sot = load_data("../experiments/model_training/SOT_influence/SOT/").reset_index()
    sot["SOT token preprended"] = True
    no_sot = load_data("../experiments/model_training/SOT_influence/no_SOT/").reset_index()
    no_sot["SOT token preprended"] = False
    sot_influence = pd.concat([sot, no_sot])
    last_epochs = sot_influence[sot_influence["epoch"] == 49]
    last_epochs = last_epochs.rename(columns={"val_sequence_accuracy": "Validation sequence accuracy"}, inplace=False)
    sns.boxplot(data=last_epochs, x="SOT token preprended", y="Validation sequence accuracy", ax=axes[0, 1])

    pad_handling = load_data("../experiments/model_training/pad_handling/").reset_index()
    last_epochs = pad_handling[pad_handling["epoch"] == 49]
    last_epochs = last_epochs.rename(columns={"val_sequence_accuracy": "Validation sequence accuracy",
                                              "fitness_ignore_pad": "Ignore padding"}, inplace=False)
    sns.boxplot(data=last_epochs, x="Ignore padding", y="Validation sequence accuracy", ax=axes[1, 0])

    pad_handling = load_data("../experiments/model_training/dim_model/").reset_index()
    last_epochs = pad_handling[pad_handling["epoch"] == 49]
    last_epochs = last_epochs.rename(columns={"val_sequence_accuracy": "Validation sequence accuracy",
                                              "dim_model": "Model dimension"}, inplace=False)
    sns.boxplot(data=last_epochs, x="Model dimension", y="Validation sequence accuracy", ax=axes[1, 1])

    plt.savefig(f"../plots/training_variants.pdf", bbox_inches='tight')


    df = load_data("../experiments/model_training/SOT_influence/SOT/505_tecator/")
    df["epoch"] = df.index

    df.rename(columns={"val_loss": "Validation loss", "train_loss": "Training loss",
                       "val_token_accuracy": "Validation token accuracy",
                       "val_sequence_accuracy": "Validation sequence accuracy"},
              inplace=True)
    df_melted = df[["Training loss", "Validation loss", "Validation token accuracy",
                    "Validation sequence accuracy", "epoch"]].melt(id_vars='epoch', var_name='metric', value_name='value')
    g = sns.FacetGrid(df_melted, col='metric', col_wrap=2, height=3, sharex=False, sharey=False)

    g.map(sns.lineplot, 'epoch', 'value')
    g.fig.set_size_inches(figsize)
    custom_labels = {
        "Validation loss": 'Loss',
        "Training loss": 'Loss',
        "Validation token accuracy": 'Accuracy',
        "Validation sequence accuracy": 'Accuracy'
    }

    # Iterate over the axes and set the custom labels
    for ax, metric in zip(g.axes.flat, g.col_names):
        ax.set_ylabel(custom_labels[metric])
    g.add_legend()

    plt.savefig(f"../plots/training_metrics.pdf", bbox_inches='tight')

def plot_semantics():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    similarity_heatmap("../experiments/model_training/SOT_influence/SOT/505_tecator/2024-06-10_05-18-18_dhcpf118.fit.vutbr.cz/model.pth",
                       "505_tecator", ignore_pad=False, SOT=True, masked=False,
                       kwargs={"ax": axes[0], "vmin": 0, "vmax": 0.05, "cbar": False})
    axes[0].set_title("Without masked learning")

    similarity_heatmap("../training-runs/2024-06-13_15-32-01_dhcpg197.fit.vutbr.cz/model.pth",
                       "505_tecator", ignore_pad=False, SOT=True, masked=True,
                       kwargs={"ax": axes[1], "vmin": 0, "vmax": 0.05, "cbar": False})
    axes[1].set_title("With masked learning")

    cb = fig.colorbar(axes[0].collections[0], ax=axes, orientation='horizontal', fraction=0.05)
    cb.set_label("Cosine similarity")

    plt.savefig("../plots/math_semantics.pdf", bbox_inches='tight')
    plt.show()


def plot_binary_semantics():
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # similarity_heatmap("../training-runs/2024-06-18_22-27-40_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=False, SOT=True, masked=False)
    # similarity_heatmap("../training-runs/2024-06-16_19-49-31_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=True, SOT=True, masked=False)
    similarity_heatmap("../training-runs/2024-06-15_16-21-40_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
                       ignore_pad=False, SOT=True, masked=True, kwargs={"ax": axes[0, 0]})
    axes[0, 0].set_title("With masked learning, last encoder state")

    similarity_heatmap("../training-runs/2024-06-16_16-32-26_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
                       ignore_pad=False, SOT=True, masked=False, kwargs={"ax": axes[0, 1]})
    axes[0, 1].set_title("Without masked learning, last encoder state")

    similarity_heatmap("../training-runs/2024-06-15_16-21-40_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
                       ignore_pad=False, SOT=True, masked=True, kwargs={"ax": axes[1, 0]}, use_decoder_embedding=True)
    axes[1, 0].set_title("With masked learning, last decoder state")

    similarity_heatmap("../training-runs/2024-06-16_16-32-26_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
                       ignore_pad=False, SOT=True, masked=False, kwargs={"ax": axes[1, 1]}, use_decoder_embedding=True)
    axes[1, 1].set_title("Without masked learning, last decoder state")

    plt.savefig("../plots/binary_semantics.pdf", bbox_inches='tight')
    plt.show()


def main():
    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-param--forceTrue-trials20")
    plot_data(df, "mut_param", "mut_param_value", None)
    plot_succes_rate(df, "mut_param", "Mutation succes rate", None)
    exit()

    plot_binary_semantics()
    #plot_semantics()
    plot_random_noise_stats()
    # plot_comparison()
    # plot_rev_euclid_dist_stats()
    # plot_euclid_dist()
    #
    # plot_random_noise()
    # plot_baseline()
    # model_plots()


    #benchmark_mutUniform_cxOnePoint()
    # benchmark_de_mut_cxOnePoint()
    # benchmark_random_noise_cxOnePoint()
    # hamming_distance_heatmap(3)
    # model_plots()
    #benchmark_rev_euclid_dist_cxOnePoint()
    #benchmark_rev_cosine_dist_cxOnePoint()

    # similarity_heatmaps()
    #model_plots()
    # similarity_heatmap("../training-runs/2024-06-18_22-27-40_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=False, SOT=True, masked=False, aggregate="sum")
    # similarity_heatmap("../training-runs/2024-06-16_19-49-31_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=True, SOT=True, masked=False, aggregate="sum")
    # similarity_heatmap("../training-runs/2024-06-15_16-21-40_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=False, SOT=True, masked=True, aggregate="sum")
    # similarity_heatmap("../training-runs/2024-06-16_16-32-26_dhcpg197.fit.vutbr.cz/model.pth", "3-parity",
    #                    ignore_pad=False, SOT=True, masked=False, aggregate="sum")

    #benchmark_mutUniform_cxOnePoint()

    # similarity_heatmap("../training-runs/2024-06-13_17-03-03_dhcpf244.fit.vutbr.cz/model.pth",
    #                    "505_tecator", ignore_pad=True, SOT=True, masked=True, aggregate="sum")

    # df = load_data("../runs/evolution/--tournament-size")
    # plot_data(df, "tournament_size")
    # df = load_data("../runs/evolution/mutUniform_cxOnePoint")
    # plot_data(df, "pop_size")

    #
    # df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-param")
    # plot_data(df, "mut_param")
    #
    #
    # df = load_data("../runs/evolution/de_mut_cxOnePoint/--mut-param/505_tecator/")
    # print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    # plot_data(df, "mut_param")
    #
    #
    # df = load_data("../runs/evolution/mut_rev_cosine_dist_cxOnePoint/--mut-param")
    # plot_data(df, "mut_param", "Distance")
    # df = load_data("../runs/evolution/--p-cross")
    # plot_data(df, "p_cross")

    # mutUniform_cxOnePoint = load_data("../runs/evolution/mutUniform_cxOnePoint/fup-check/505_tecator/--tournament-size-7")
    # de_mut_cxOnePoint = load_data("../runs/evolution/fucked_num_evaluations/de_mut_cxOnePoint/--tournament-size/505_tecator/--tournament-size-7")
    # plot_convergence({"mutUniform_cxOnePoint": mutUniform_cxOnePoint,
    #                        "de_mut_cxOnePoint": de_mut_cxOnePoint},
    #                  "Generation", "Fitness", "../plots/de_mut_cxOnePoint_convergence.pdf", visualize=True)


if __name__ == "__main__":
    main()
