from pprint import pprint

from numpy import hstack

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39
WINDOWS_BEFORE_CRASH_TO_ANALISE = 6


def _calc_precision_recall_f1_fpr(
    windows_TP, windows_FN, windows_FP, windows_TN
):
    TP_FN = windows_TP + windows_FN
    TP_FP = windows_TP + windows_FP

    if (
        TP_FP == 0
    ):  # nominal case (only TP e FP are 0, no anomalies) #TODO: check if its good to return 0
        precision = 0
    else:
        precision = windows_TP / TP_FP

    if (
        TP_FN == 0
    ):  # nominal case (only positive example, no negative) #TODO: check if its good to return 0
        recall = 0
    else:
        recall = windows_TP / TP_FN

    p_r = precision + recall
    if p_r == 0:  # nominal case #TODO: check if its good to return 0
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / p_r)

    FP_TN = windows_FP + windows_TN
    if FP_TN == 0:  # nominal case #TODO: check if its good to return 0
        fpr = 0
    else:
        fpr = windows_FP / FP_TN
    return precision, recall, f1_score, fpr


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
    tot_window_TP, tot_window_FN, tot_window_FP, tot_window_TN = (
        0,
        0,
        0,
        0,
    )  # EACH FRAME
    window_TP, window_FN, window_FP, window_TN = 0, 0, 0, 0  # EACH WINDOW
    window_before_crash = []

    for j in range(len(window)):
        current_frame = (window_number * NORMAL_WINDOW_LENGTH) + j
        if (
            crashes_per_frame.get(current_frame) == 1
        ):  # window with crash separated in 2 array: before/after (of original window)
            window_crash = True
            window_before_crash = window[0:j]
            tot_window_TP, tot_window_FN = get_window_positive_negative(
                window_before_crash, threshold
            )
            if tot_window_TP > tot_window_FN:
                window_TP = 1
            elif tot_window_TP < tot_window_FN:
                window_FN = 1
            break

    if len(window_before_crash) > 0:  # analysis window with crash
        window_after_crash = window[len(window_before_crash) :]
        tot_window_FP, tot_window_TN = get_window_positive_negative(
            window_after_crash, threshold
        )
    else:
        window_crash = False
        tot_window_FP, tot_window_TN = get_window_positive_negative(
            window, threshold
        )

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

    for i in range(len(uncertainties_windows)):
        (
            window_crash,
            window_TP,
            window_FN,
            window_FP,
            window_TN,
        ) = window_analysis(
            i, uncertainties_windows[i], crashes_per_frame, threshold
        )
        windows.append(
            {
                "window": uncertainties_windows[i],
                "start_frame": int(i * NORMAL_WINDOW_LENGTH),
                "end_frame": int(
                    (((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1
                ),
                "is_crash": int(window_crash),
                "FP": window_FP,
                "TN": window_TN,
                "TP": window_TP,
                "FN": window_FN,
            }
        )  # based on windows DB-schema columns

    (
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    ) = (0, 0, 0, 0, 0)
    for row in windows:
        tot_windows_TP += row.get("TP")
        tot_windows_FN += row.get("FN")
        tot_windows_FP += row.get("FP")
        tot_windows_TN += row.get("TN")
        tot_crashes += row.get("is_crash")

    return (
        windows,
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    )


def before_window_crash_analysis(
    uncertainties_windows, current_frame, threshold
):
    windows_before_crash = []
    tot_window_FP, tot_window_TN = 0, 0
    window_FP, window_TN = 0, 0

    uncertainties_windows_flatten = list(uncertainties_windows.flatten())

    frame_before_crash = int(current_frame - NORMAL_WINDOW_LENGTH)
    total_frames_before_crash = int(
        NORMAL_WINDOW_LENGTH * WINDOWS_BEFORE_CRASH_TO_ANALISE
    )
    first_frame_of_window_series = int(
        frame_before_crash - total_frames_before_crash
    )
    # print("\t\tCrash Frame: " + str(current_frame) +"\n\t\tFrame before crash frame: " + str(frame_before_crash) + "\n\t\tFirst frame of window series: " + str(first_frame_of_window_series))

    if frame_before_crash < 39:
        print("Crash occurs in first window, can't analyze backwards...")
    else:
        for i in range(
            first_frame_of_window_series,
            frame_before_crash,
            NORMAL_WINDOW_LENGTH,
        ):
            # i will increment from _frame_before_windows_crash (6s before start of the window_crash) to start_frame (start of the window crash)
            end_frame_x_window = (i + NORMAL_WINDOW_LENGTH) - 1
            # print("\t\t\tWindow start frame: " + str(i) + "\n\t\t\tWindow end frame: " + str(end_frame_x_window) + "\n\t\t\ti: " + str(i))
            x_window_before_crash = uncertainties_windows_flatten[
                i:end_frame_x_window
            ]
            tot_window_FP, tot_window_TN = get_window_positive_negative(
                x_window_before_crash, threshold
            )

            if tot_window_FP > tot_window_TN:
                window_FP = 1
            elif tot_window_FP < tot_window_TN:
                window_TN = 1

            windows_before_crash.append(
                {
                    "window": list(x_window_before_crash),
                    "start_frame": int(i),
                    "end_frame": int(end_frame_x_window),
                    "is_crash": 0,
                    "FP": window_FP,
                    "TN": window_TN,
                    "TP": 0,
                    "FN": 0,
                }
            )

    return windows_before_crash


def window_crash_analysis(uncertainties_windows, current_frame, threshold):
    start_frame = current_frame - NORMAL_WINDOW_LENGTH
    uncertainties_windows_flatten = list(uncertainties_windows.flatten())
    window_before_crash = uncertainties_windows_flatten[
        start_frame:current_frame
    ]

    window_TP, window_FN = get_window_positive_negative(
        window_before_crash, threshold
    )

    return window_before_crash, window_TP, window_FN


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
    window_TP, window_FN, window_FP, window_TN = 0, 0, 0, 0
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
        windows_before_crash = before_window_crash_analysis(
            uncertainties_windows, frame, threshold
        )

        # pprint(windows_before_crash)
        windows.extend(windows_before_crash)

        window, tot_window_TP, tot_window_FN = window_crash_analysis(
            uncertainties_windows, frame, threshold
        )

        if tot_window_TP > tot_window_FN:
            window_TP = 1
        elif tot_window_TP < tot_window_FN:
            window_FN = 1

        windows.append(
            {
                "window": window,
                "start_frame": int(frame - NORMAL_WINDOW_LENGTH),
                "end_frame": int(frame - 1),
                "is_crash": 1,
                "TP": window_TP,
                "FN": window_FN,
                "FP": 0,
                "TN": 0,
            }
        )

    print(
        ">> Analyzed windows (crash windows + 6 before every crash window): ",
        len(windows),
    )

    for row in windows:
        tot_windows_FP += int(row.get("FP"))
        tot_windows_TN += int(row.get("TN"))
        tot_windows_TP += int(row.get("TP"))
        tot_windows_FN += int(row.get("FN"))
        tot_crashes += int(row.get("is_crash"))

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
    window_TP, window_FN = (0, 0)
    tot_window_TP, tot_window_FN = (0, 0)
    crashes_frames = []

    for i in range(len(uncertainties_windows)):
        for j in range(len(uncertainties_windows[i])):
            current_frame = (i * NORMAL_WINDOW_LENGTH) + j
            if (crashes_per_frame.get(current_frame) == 1) and (
                crashes_per_frame.get(current_frame - 1) == 0
            ):
                window_crash = True
                window, tot_window_TP, tot_window_FN = window_crash_analysis(
                    uncertainties_windows, current_frame, threshold
                )

                if tot_window_TP > tot_window_FN:
                    window_TP = 1
                elif tot_window_TP < tot_window_FN:
                    window_FN = 1

                windows.append(
                    {
                        "window": window,
                        "start_frame": int(
                            current_frame - NORMAL_WINDOW_LENGTH
                        ),
                        "end_frame": int(
                            (
                                (
                                    (current_frame - NORMAL_WINDOW_LENGTH)
                                    + NORMAL_WINDOW_LENGTH
                                )
                                - 1
                            )
                        ),
                        "is_crash": int(window_crash),
                        "TP": window_TP,
                        "FN": window_FN,
                    }
                )  # based on windows DB-schema columns

                (
                    tot_windows_TP,
                    tot_windows_FN,
                    tot_crashes,
                ) = (0, 0, 0)

    for row in windows:
        tot_windows_TP += row.get("TP")
        tot_windows_FN += row.get("FN")
        tot_crashes += row.get("is_crash")

    print(">> Crashes Found: " + str(tot_crashes))

    return (windows, tot_windows_TP, tot_windows_FN, tot_crashes)


def main():
    print("This script is not intended to be run directly.")


if __name__ == "__main__":
    main()
