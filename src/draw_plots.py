import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_evolution_results(results: pd.DataFrame, output_dir: str):
    pass


def add_property(folder: str, property: str, value: float):
    pathlist = list(Path(folder).rglob('*.pkl'))
    for i, path in enumerate(pathlist):
        print(f"Altering {i + 1} / {len(pathlist)}: {path}")
        df = pd.read_pickle(path)
        df[property] = value
        df.to_pickle(path)


def load_data(folder: str = "../runs/evolution/--p-cross"):
    pathlist = list(Path(folder).rglob('*.pkl'))
    dataframes = []
    for i, path in enumerate(pathlist):
        print(f"Loading {i + 1} / {len(pathlist)}: {path} \r", end="")
        dataframes.append(pd.read_pickle(path))

    return pd.concat(dataframes)


def plot_data(df: pd.DataFrame, feature: str):
    grouped = df.groupby([feature.strip("-").replace("-", "_"), "run_id"]).agg({"fit-min": "min"})
    sns.boxplot(grouped.reset_index(), y="fit-min", x=feature)
    plt.show()


def main():

    # df = load_data("../runs/evolution/--tournament-size")
    # plot_data(df, "tournament_size")
    # df = load_data("../runs/evolution/mutUniform_cxOnePoint")
    # plot_data(df, "pop_size")

    df = load_data("../runs/evolution/mut_add_random_noise_gaussian_cxOnePoint/--mut-param")
    plot_data(df, "mut_param")

    # df = load_data("../runs/evolution/mut_rev_cosine_dist_cxOnePoint/--mut-param")
    # plot_data(df, "mut_param")
    # df = load_data("../runs/evolution/--p-cross")
    # plot_data(df, "p_cross")

    # df = load_data("../runs/evolution/--p-mut")
    # plot_data(df, "p_mut")

    # df = load_data("../runs/evolution/--pop-size")
    # plot_data(df, "pop_size")

    # #grouped = df.groupby(["dataset", "pop_size", "run_id"]).agg({"fit-min": "min"})
    # grouped = df.groupby(["pop_size", "run_id"]).agg({"fit-min": "min"})
    # indexed = df.set_index(["dataset", "pop_size"])
    # reseted = indexed.reset_index()
    # sns.boxplot(reseted, y="fit-min", x="p_cross")
    # plt.show()
    # print(df)


if __name__ == "__main__":
    main()
