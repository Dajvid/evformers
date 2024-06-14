import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmlb
import seaborn as sns
from pathlib import Path

import torch

from gp.Pset import create_basic_symreg_pset
from gp.sym_reg_tree import SymRegTree, get_mapping
from gpformer.model import Transformer


def plot_evolution_results(results: pd.DataFrame, output_dir: str):
    pass


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
    plt.figure()
    df.rename(columns={"fit-min": "fitness", feature: x_label}, inplace=True)
    grouped = df.groupby([x_label, "run_id"]).agg({"fitness": "min"})
    sns.boxplot(grouped.reset_index(), y="fitness", x=x_label)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def similarity_heatmap(model_weights="../model-tecator-0-6-SOT.pth", dataset="505_tecator", SOT=True, masked=False,
                       measure="cos_dist", ignore_pad=False, aggregate=None):
    pset = create_basic_symreg_pset(pmlb.fetch_data(dataset))
    extra_elements = ["PAD", "UNKNOWN"]
    if SOT:
        extra_elements.append("SOT")
    mapping = get_mapping(pset, extra_elements) if SOT else get_mapping(pset, ["PAD", "UNKNOWN"])
    if masked:
        max_token_id = max(list(mapping.values()))
        mapping["MASK"] = max_token_id + 1

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
    model = Transformer(mapping, 2 * 6, 2, 1, 1,
                        6, 2, ignore_pad=ignore_pad).to("cpu")
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))

    trees = {key: tree.embedding(model, "cpu", 6, mapping) for key, tree in trees.items()}

    matrix = np.zeros((len(trees), len(trees)))
    for i, (key1, tree1) in enumerate(trees.items()):
        for j, (key2, tree2) in enumerate(trees.items()):
            if i == j:
                matrix[i, j] = 0
            else:
                if measure == "cos_dist":
                    if aggregate == "sum":
                        matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(tree1.sum(dim=0), tree2.sum(dim=0), dim=0)
                    else:
                        matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(tree1.flatten(), tree2.flatten(), dim=0)
                else:
                    matrix[i, j] = torch.nn.functional.mse_loss(tree1, tree2)

    sns.heatmap(matrix, xticklabels=trees.keys(), yticklabels=trees.keys())
    plt.show()


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
    # df = load_data("../runs/evolution/de_mut_cxOnePoint/--mut-param")
    # print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    # plot_data(df, "mut_param", "F", "../plots/de_mut_cxOnePoint_mut_param.pdf")
    #
    # df = load_data("../runs/evolution/de_mut_cxOnePoint/--pop-size")
    # print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    # plot_data(df, "pop_size", "Population size", "../plots/de_mut_cxOnePoint_pop_size.pdf")
    #
    # df = load_data("../runs/evolution/de_mut_cxOnePoint/--mut-ratio-param")
    # print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    # plot_data(df, "mut_ratio_param", "CR", "../plots/de_mut_cxOnePoint_mut_ratio_param.pdf")

    # "../plots/de_mut_cxOnePoint_p_mut.pdf"
    df = load_data("../runs/evolution/de_mut_cxOnePoint/--p-mut")
    print("Mutation succes rate: ", df["mut_success"].sum() / df["mut_called"].sum())
    plot_data(df, "p_mut", "Mutation probability")


def model_plots():

    df = load_data("../experiments/model_training/SOT_influence/SOT/505_tecator/")
    df["epoch"] = df.index

    df_melted = df[["train_loss", "val_loss", "val_token_accuracy",
                    "val_sequence_accuracy", "epoch"]].melt(id_vars='epoch', var_name='metric', value_name='value')
    g = sns.FacetGrid(df_melted, col='metric', col_wrap=2, height=3, sharex=False, sharey=False, aspect=1.5)

    g.map(sns.lineplot, 'epoch', 'value')

    custom_labels = {
        'val_loss': 'Loss',
        'train_loss': 'Loss',
        'val_token_accuracy': 'Accuracy',
        'val_sequence_accuracy': 'Accuracy'
    }

    # Iterate over the axes and set the custom labels
    for ax, metric in zip(g.axes.flat, g.col_names):
        ax.set_ylabel(custom_labels[metric])
    g.add_legend()

    plt.savefig(f"../plots/training_metrics.pdf")


def main():
    # similarity_heatmaps()
    #model_plots()
    benchmark_de_mut_cxOnePoint()
    # similarity_heatmap("../training-runs/2024-06-13_15-32-01_dhcpg197.fit.vutbr.cz/model.pth",
    #                    "505_tecator", ignore_pad=False, SOT=True, masked=True, aggregate="sum")
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
    # plot_data(df, "mut_param")
    # df = load_data("../runs/evolution/--p-cross")
    # plot_data(df, "p_cross")


if __name__ == "__main__":
    main()
