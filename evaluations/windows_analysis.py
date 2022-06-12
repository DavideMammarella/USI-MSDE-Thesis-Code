from pprint import pprint

from numpy import hstack

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39
WINDOWS_BEFORE_CRASH_TO_ANALISE = 6


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

def before_window_crash_analysis(
        uncertainties_windows, current_frame, threshold
) -> dict:
    window_FP, window_TN = 0, 0

    uncertainties_windows_flatten = list(uncertainties_windows.flatten())
    total_frames_before_crash = int(
        NORMAL_WINDOW_LENGTH * WINDOWS_BEFORE_CRASH_TO_ANALISE
    )
    # print("\t\tCrash Frame: " + str(current_frame) +"\n\t\tFirst frame of window crash: " + str(frame_before_crash) + "\n\t\tFirst frame of window series: " + str(first_frame_of_window_series))

    if current_frame < 39:
        print("Crash occurs in first window, can't analyze backwards...\nInstead first window is analyzed...")
        start_frame = 0
        end_frame = NORMAL_WINDOW_LENGTH - 1
    else:
        start_frame_window_crash = int((current_frame - 1) - NORMAL_WINDOW_LENGTH)
        start_frame = int((start_frame_window_crash - 1) - total_frames_before_crash)
        end_frame = (start_frame + NORMAL_WINDOW_LENGTH - 1)
        # print("\t\t\tWindow start frame: " + str(first_frame_of_window_series) + "\n\t\t\tWindow end frame: " + str(end_frame_x_window))

    window_before_crash = uncertainties_windows_flatten[start_frame:end_frame]
    tot_window_FP, tot_window_TN = get_window_positive_negative(
        window_before_crash, threshold
    )

    if tot_window_FP > tot_window_TN:
        window_FP = 1
    elif tot_window_FP < tot_window_TN:
        window_TN = 1
    elif tot_window_FP == tot_window_TN:
        window_TN = 1

    return {
        "window": list(window_before_crash),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "is_crash": int(0),
        "FP": int(window_FP),
        "TN": int(window_TN),
        "TP": int(0),
        "FN": int(0)
    }


def window_crash_analysis(uncertainties_windows, current_frame, threshold) -> dict:
    window_TP, window_FN = 0, 0
    start_frame = current_frame - NORMAL_WINDOW_LENGTH - 1
    uncertainties_windows_flatten = list(uncertainties_windows.flatten())
    crash_window = uncertainties_windows_flatten[
                   start_frame:current_frame
                   ]

    tot_window_TP, tot_window_FN = get_window_positive_negative(
        crash_window, threshold
    )

    if tot_window_TP > tot_window_FN:
        window_TP = 1
    elif tot_window_TP < tot_window_FN:
        window_FN = 1
    elif tot_window_TP == tot_window_FN:
        window_TP = 1

    return {
        "window": list(crash_window),
        "start_frame": int(start_frame),
        "end_frame": int(current_frame),
        "is_crash": int(1),
        "TP": int(window_TP),
        "FN": int(window_FN),
        "FP": int(0),
        "TN": int(0)
    }

def window_nominal_analysis(i, window,threshold):
    assert len(window) == NORMAL_WINDOW_LENGTH

    window_crash = False
    window_FP, window_TN, tot_window_FP, tot_window_TN = 0, 0, 0, 0

    tot_window_FP, tot_window_TN = get_window_positive_negative(
            window, threshold
        )

    if tot_window_FP > tot_window_TN:
        window_FP = 1
    elif tot_window_FP < tot_window_TN:
        window_TN = 1
    elif tot_window_FP == tot_window_TN:
        window_TN = 1

    return {
                "window": list(window),
                "start_frame": int(i * NORMAL_WINDOW_LENGTH),
                "end_frame": int(
                    (((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1
                ),
                "is_crash": int(window_crash),
                "FP": int(window_FP),
                "TN": int(window_TN),
                "TP": int(0),
                "FN": int(0),
            }


def _on_windows_nominal(uncertainties_windows, crashes_per_frame, threshold):
    windows = []

    for i in range(len(uncertainties_windows)):
        (
            window
        ) = window_nominal_analysis(
            i, uncertainties_windows[i], threshold
        )
        windows.append(window)  # based on windows DB-schema columns

    return windows


def _on_anomalous_alternative(
        uncertainties_windows, crashes_per_frame, threshold
):
    (
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    ) = (0, 0, 0, 0, 0)
    crashes_frames = []
    windows = []

    for i in range(len(uncertainties_windows)):
        for j in range(len(uncertainties_windows[i])):
            current_frame = (i * NORMAL_WINDOW_LENGTH) + j
            if (crashes_per_frame.get(current_frame) == 1) and (
                    crashes_per_frame.get(current_frame - 1) == 0
            ):
                crashes_frames.append(current_frame)

    for frame in crashes_frames:
        window_before_crash = before_window_crash_analysis(
            uncertainties_windows, frame, threshold
        )

        # pprint(windows_before_crash)
        windows.append(window_before_crash)

        crash_window = window_crash_analysis(
            uncertainties_windows, frame, threshold
        )

        windows.append(crash_window)

    #pprint(windows)
    for row in windows:
        tot_windows_FP += int(row.get("FP"))
        tot_windows_TN += int(row.get("TN"))
        tot_windows_TP += int(row.get("TP"))
        tot_windows_FN += int(row.get("FN"))
        tot_crashes += int(row.get("is_crash"))

    print(">> Crashes Found: " + str(tot_crashes))
    print(
        ">> Analyzed windows (crash windows + 6th window before crash windows): ",
        len(windows),
    )

    assert (tot_windows_TP + tot_windows_FN + tot_windows_FP + tot_windows_TN) == len(windows)

    return (
        windows,
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    )


def _on_anomalous(uncertainties_windows, crashes_per_frame, threshold):
    windows = []
    tot_windows_TP, tot_windows_FN, tot_crashes = (0, 0, 0)
    crashes_frames = []

    for i in range(len(uncertainties_windows)):
        for j in range(len(uncertainties_windows[i])):
            current_frame = (i * NORMAL_WINDOW_LENGTH) + j
            if (crashes_per_frame.get(current_frame) == 1) and (
                    crashes_per_frame.get(current_frame - 1) == 0
            ):
                crashes_frames.append(current_frame)

    for frame in crashes_frames:
        crash_window = window_crash_analysis(
            uncertainties_windows, frame, threshold
        )

        windows.append(crash_window)

    for row in windows:
        tot_windows_TP += int(row.get("TP"))
        tot_windows_FN += int(row.get("FN"))
        tot_crashes += int(row.get("is_crash"))

    print(">> Crashes Found: " + str(tot_crashes))
    print(
        ">> Analyzed windows (crash windows): ",
        len(windows),
    )

    assert (tot_windows_TP + tot_windows_FN) == len(windows)

    return (windows, tot_windows_TP, tot_windows_FN, tot_crashes)


def main():
    print("This script is not intended to be run directly.")


if __name__ == "__main__":
    main()
