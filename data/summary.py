from pathlib import Path
from pprint import pprint

import pandas as pd

from utils import navigate


def calculate_results(df):
    return df.groupby("threshold_type", as_index=False).agg(
        TP=("true_positives", "sum"),
        FP=("false_positives", "sum"),
        TN=("true_negatives", "sum"),
        FN=("false_negatives", "sum"),
        Prec=("prec", "mean"),
        Recall=("recall", "mean"),
        F1=("f1", "mean"),
    )


def normalize_dataframe(df):
    return df.drop(
        columns=["simulation", "num_normal", "auroc", "false_positive_rate", "pr_auc"]
    )


def get_simulator(csv_file, simulator_name):
    iter_csv = pd.read_csv(csv_file, iterator=True, chunksize=1000)
    df = pd.concat(
        [chunk[chunk["simulation"].str.contains(simulator_name)] for chunk in iter_csv]
    )
    return df


def collect_analysis(metrics_path):
    analysis_names = []

    for analysis in metrics_path.iterdir():
        if analysis.is_file() and analysis.name.endswith(".csv"):
            analysis_names.append(analysis.name)

    return analysis_names


def main():
    cfg = navigate.config()
    metrics_path = navigate.performance_metrics_dir()
    sims = ["xai", "DAVE2"]

    analysis_names = collect_analysis(metrics_path)

    for i, name in enumerate(analysis_names, start=1):
        csv_file = Path(metrics_path, name)
        print("\n###########################################################")
        print("ANALYZING  " + str(name))
        print("###########################################################")

        for sim in sims:
            print("\n>> Simulator: " + sim)
            sim_df = get_simulator(csv_file, sim)
            sim_df = normalize_dataframe(sim_df)
            result_df = calculate_results(sim_df)
            print(result_df)
            result_df.to_csv(
                Path(metrics_path, "analysis_" + str(i) + "_summary_" + sim + ".csv"),
                index=False,
            )


if __name__ == "__main__":
    main()
