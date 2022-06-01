NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30

def _on_anomalous(nominal_uncertainties, treshold):
    windows = []

    window_TP = 0
    window_FN = 0
    tot_TP = 0
    tot_FN = 0

    for i in range(len(nominal_uncertainties)): # ROW (window)
        for j in range(len(nominal_uncertainties[i])): # COLUMN (uncertainties inside window)
            if nominal_uncertainties[i][j] > treshold:
                window_FP += 1
            else:
                window_TN += 1

        assert len(nominal_uncertainties[i]) == NORMAL_WINDOW_LENGTH

        if window_FP > window_TN:
            tot_FP += 1
        else:
            tot_TN += 1

        windows.append({
            "window_id": int(i),
            "start_frame": int(i * NORMAL_WINDOW_LENGTH),
            "end_frame": int((((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1),
            "FP": window_FP,
            "TN": window_TN
        })  # based on windows DB-schema columns
        window_FP = 0
        window_TN = 0

    print("----------------------------------------------------------")
    print("Analyzing with THRESHOLD: " + str(treshold))
    print(">> Windows Analyzed: " + str(len(nominal_uncertainties)))
    print(">> TP: " + str(tot_FP))
    print(">> TN: " + str(tot_TN))
    #for row in windows:
    #    print("Window: " + str(row["window_id"]) + "\nFP: " + str(row["FP"]) + "\nTN: " + str(row["TN"]))

    return windows, tot_FP, tot_TN

def _on_nominal(nominal_uncertainties, treshold):
    windows = []

    window_FP = 0
    window_TN = 0
    tot_FP = 0
    tot_TN = 0

    for i in range(len(nominal_uncertainties)): # ROW (window)
        for j in range(len(nominal_uncertainties[i])): # COLUMN (uncertainties inside window)
            if nominal_uncertainties[i][j] > treshold:
                window_FP += 1
            else:
                window_TN += 1

        assert len(nominal_uncertainties[i]) == NORMAL_WINDOW_LENGTH

        if window_FP > window_TN:
            tot_FP += 1
        else:
            tot_TN += 1

        windows.append({
            "window_id": int(i),
            "start_frame": int(i * NORMAL_WINDOW_LENGTH),
            "end_frame": int((((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1),
            "FP": window_FP,
            "TN": window_TN
        })  # based on windows DB-schema columns
        window_FP = 0
        window_TN = 0

    print("----------------------------------------------------------")
    print("Analyzing with THRESHOLD: " + str(treshold))
    print(">> Windows Analyzed: " + str(len(nominal_uncertainties)))
    print(">> TP: " + str(tot_FP))
    print(">> TN: " + str(tot_TN))
    #for row in windows:
    #    print("Window: " + str(row["window_id"]) + "\nFP: " + str(row["FP"]) + "\nTN: " + str(row["TN"]))

    return windows, tot_FP, tot_TN

def main():
    print("This script is not intended to be run directly.")

if __name__ == "__main__":
    main()