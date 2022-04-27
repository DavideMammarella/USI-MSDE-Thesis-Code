# !!! Simulations must be in the project folder under: simulations/<name_of_simulation>
# !!! <name_of_simulation> folder must contain: IMG folder (with .jpg) and driving_log.csv

import os
from pathlib import Path


def visit_and_collect_simulations(curr_project_path):
    """
    Visit all simulation folders and collect only the names of simulations not previously analysed ("-uncertainty-evaluated").
    :return: list of simulations
    """
    sims_path = Path(curr_project_path, "simulations")

    # First Iteration: collect all simulations
    _, dirs, _ = next(os.walk(sims_path))  # list all folders in simulations_path (only top level)

    # Second iteration: collect all simulations to exclude
    exclude = []
    for d in dirs:
        if "-uncertainty-evaluated" in d:
            exclude.append(d)
            exclude.append(d[:-len("-uncertainty-evaluated")])

    sims_evaluated = int(len(exclude)/2)
    print("Total simulations:\t", len(dirs)-sims_evaluated)
    print("Simulations already evaluated:\t", sims_evaluated)

    # Third iteration: collect all simulations to evaluate (excluding those already evaluated)
    sims = [d for d in dirs if d not in exclude]
    print("Simulations to evaluate:\t", len(sims))

    return sims


def main():
    curr_project_path = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))  # overcome OS issues

    sims = visit_and_collect_simulations(curr_project_path)


if __name__ == "__main__":
    main()
