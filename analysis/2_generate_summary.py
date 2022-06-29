from pathlib import Path

import pandas as pd

from utils.navigate import collect_analysis_results, results_dir

METRIC_TO_EVAL = "loss"  # unc/loss
SIMS = ["xai", "DAVE2"]


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


def main():
    results_path = Path(results_dir(), METRIC_TO_EVAL)
    analysis_names = collect_analysis_results(results_path)

    for i, name in enumerate(analysis_names, start=1):
        csv_file = Path(results_path, name)
        print("\n###########################################################")
        print("ANALYZING  " + "/".join(str(csv_file).rsplit("/")[-2:]))
        print("###########################################################")

        for sim in SIMS:
            print("\n>> Simulator: " + sim)
            sim_df = get_simulator(csv_file, sim)
            sim_df = normalize_dataframe(sim_df)
            result_df = calculate_results(sim_df)
            print(result_df)
            result_df.to_csv(
                Path(results_path, "analysis_" + str(i) + "_summary_" + sim + ".csv"),
                index=False,
            )


if __name__ == "__main__":
    main()
