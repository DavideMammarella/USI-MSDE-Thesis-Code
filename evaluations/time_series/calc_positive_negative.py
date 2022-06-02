NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 30, 30


def get_window_positive_negative(window, threshold):
    """
    Since Positive and Negative are calculated with same logic, this function is used to calculate both.
    REQUIREMENT: correct window must be passed (before crash, after crash, or without crash)
    """
    FP_or_TP, FN_or_TN = 0, 0

    for j in window:
        if j > threshold:
            FP_or_TP += 1
        else:
            FN_or_TN += 1

    return FP_or_TP, FN_or_TN


def window_analysis(window_number, window, crashes_per_frame, threshold):
    """
    windows -> 2D np array of windows (each row is a window, here expressed as: window_number)
    window -> [_,_,_,_,_,_,_] (index given by j)
    current_frame = (window_number * NORMAL_WINDOW_LENGTH) + j
    crashes_per_frame -> { frame_id : crashed }

    A window contain crash if the "current_frame" is a crash so if "current_frame" in "crashes_per_frame" ("current_frame"=="frame_id" in the dict) have value 1
    In this case the original window will be broken in 2 separate windows: before and after the crash

    TP, FN -> calculated ONLY on the BEFORE window
    FP, TN -> calculated ONLY on the AFTER window (or windows without crash)

    tot_window_... -> are TP/FN/FP/TN for each frame in the window
    window_... -> are TP/FN/FP/TN for each window (i.e. if window 1 have tot_windowFP > tot_window_TN, then window 1 is a FP)
    """
    assert len(window) == NORMAL_WINDOW_LENGTH

    window_crash = False
    tot_window_TP, tot_window_FN, tot_window_FP, tot_window_TN = 0, 0, 0, 0  # EACH FRAME
    window_TP, window_FN, window_FP, window_TN = 0, 0, 0, 0  # EACH WINDOW
    window_before_crash = []

    for j in range(len(window)):
        current_frame = (window_number * NORMAL_WINDOW_LENGTH) + j
        if crashes_per_frame.get(
                current_frame) == 1:  # window with crash separated in 2 array: before/after (of original window)
            window_crash = True
            window_before_crash = window[0:j]
            tot_window_TP, tot_window_FN = get_window_positive_negative(window_before_crash, threshold)
            if tot_window_TP > tot_window_FN:
                window_TP = 1
            elif tot_window_TP < tot_window_FN:
                window_FN = 1
            break

    if len(window_before_crash) > 0:  # analysis window with crash
        window_after_crash = window[len(window_before_crash):]
        tot_window_FP, tot_window_TN = get_window_positive_negative(window_after_crash, threshold)
    else:
        window_crash = False
        tot_window_FP, tot_window_TN = get_window_positive_negative(window, threshold)

    if tot_window_FP > tot_window_TN:
        window_FP = 1
    elif tot_window_FP < tot_window_TN:
        window_TN = 1

    return window_crash, window_TP, window_FN, window_FP, window_TN


def _on_windows(uncertainties_windows, crashes_per_frame, threshold):
    """
    window_... -> equivalent to window_... from @window_analysis
    tot_windows_... -> are sum of TP/FN/FP/TN windows (NOT each frame)
    """
    windows = []

    print("----------------------------------------------------------")
    print("Analyzing with THRESHOLD: " + str(threshold))
    print(">> Windows Analyzed: " + str(len(uncertainties_windows)))

    for i in range(len(uncertainties_windows)):
        window_crash, window_TP, window_FN, window_FP, window_TN = window_analysis(i, uncertainties_windows[i],
                                                                                   crashes_per_frame, threshold)
        windows.append({
            "window_id": int(i),
            "window": uncertainties_windows[i],
            "start_frame": int(i * NORMAL_WINDOW_LENGTH),
            "end_frame": int((((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1),
            "is_crash": int(window_crash),
            "FP": window_FP,
            "TN": window_TN,
            "TP": window_TP,
            "FN": window_FN
        })  # based on windows DB-schema columns

    tot_windows_TP, tot_windows_FN, tot_windows_FP, tot_windows_TN = 0, 0, 0, 0
    for row in windows:
        tot_windows_TP += row.get("TP")
        tot_windows_FN += row.get("FN")
        tot_windows_FP += row.get("FP")
        tot_windows_TN += row.get("TN")

    return windows, tot_windows_TP, tot_windows_FN, tot_windows_FP, tot_windows_TN


def main():
    print("This script is not intended to be run directly.")


if __name__ == "__main__":
    main()
