
NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30

def _on_nominal(nominal_uncertainties, treshold):
    windows = []

    FP = 0
    TN = 0

    for i in range(len(nominal_uncertainties)): # ROW
        for j in range(len(nominal_uncertainties[i])): # COLUMN
            if nominal_uncertainties[i][j] > treshold:
                FP += 1
            else:
                TN += 1
        assert len(nominal_uncertainties[i]) == NORMAL_WINDOW_LENGTH
        windows.append({
            "window_id": int(i),
            "start_frame": int(i * NORMAL_WINDOW_LENGTH),
            "end_frame": int((((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH))-1),
            "FP": FP,
            "TN": TN
        })

    tot_FP = FP
    tot_TN = TN

    print("----------------------------------------------------------")
    print("Analyzing with THRESHOLD: " + str(treshold))
    print("Total FP: " + str(tot_FP) + "\nTotal TN: " + str(tot_TN))
    #print(windows)
    print(">> Frame Analyzed: " + str(len(windows)))
    print(">> Windows Analyzed: " + str(len(windows)))

def main():
    print("This script is not intended to be run directly.")

if __name__ == "__main__":
    main()