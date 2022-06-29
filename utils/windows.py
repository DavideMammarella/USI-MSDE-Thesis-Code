import numpy as np

NORMAL_WINDOW_LENGTH, ANOMALY_WINDOW_LENGTH = 39, 39
WINDOWS_BEFORE_CRASH_TO_ANALISE = 6


def get_frame_ids(np_array):
    print("{}, {}".format("frame_id", "uncertainty"))
    for index, val in np.ndenumerate(np_array):
        print("{}, {}".format(index[0], val))


def windows_check(len_uncertainties, len_uncertainties_windows):
    actual_len = len_uncertainties_windows
    expected_len = len_uncertainties / NORMAL_WINDOW_LENGTH
    return int(expected_len) == int(actual_len)


def create_windows_stack(
        a: np.array, stepsize=NORMAL_WINDOW_LENGTH, width=NORMAL_WINDOW_LENGTH
):
    return np.hstack(a[i: 1 + i - width or None: stepsize] for i in range(0, width))


def get_window_positive_negative(window, threshold):
    """
    Since Positive and Negative are calculated with same logic, this function is used to calculate both.
    """
    FP_or_TP, FN_or_TN = 0, 0

    for j in window:
        if j > threshold:
            FP_or_TP += 1
        else:
            FN_or_TN += 1

    return FP_or_TP, FN_or_TN


def label_normal_window(tot_window_FP, tot_window_TN):
    window_FP, window_TN = 0, 0
    if tot_window_FP > 0:
        window_FP = 1
    else:
        window_TN = 1
    return window_FP, window_TN


def label_crash_window(tot_window_TP, tot_window_FN):
    window_TP, window_FN = 0, 0
    if tot_window_TP > 0:
        window_TP = 1
    else:
        window_FN = 1
    return window_TP, window_FN


def get_window_before_crash(uncertainties_windows, current_frame, threshold) -> dict:
    window_FP, window_TN = 0, 0

    uncertainties_windows_flatten = list(uncertainties_windows.flatten())
    total_frames_before_crash = int(
        NORMAL_WINDOW_LENGTH * WINDOWS_BEFORE_CRASH_TO_ANALISE
    )

    if current_frame < 39:
        print(
            "Crash occurs in first window, can't analyze backwards...\nInstead first window is analyzed..."
        )
        start_frame = 0
        end_frame = NORMAL_WINDOW_LENGTH - 1
    else:
        start_frame_window_crash = int((current_frame - 1) - NORMAL_WINDOW_LENGTH)
        start_frame = int((start_frame_window_crash - 1) - total_frames_before_crash)
        end_frame = start_frame + NORMAL_WINDOW_LENGTH - 1

    window_before_crash = uncertainties_windows_flatten[start_frame:end_frame]
    tot_window_FP, tot_window_TN = get_window_positive_negative(
        window_before_crash, threshold
    )

    window_FP, window_TN = label_normal_window(tot_window_FP, tot_window_TN)

    return {
        "window": list(window_before_crash),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "is_crash": int(0),
        "FP": int(window_FP),
        "TN": int(window_TN),
        "TP": int(0),
        "FN": int(0),
    }


def get_crash_window(uncertainties_windows, current_frame, threshold) -> dict:
    start_frame = current_frame - NORMAL_WINDOW_LENGTH - 1
    uncertainties_windows_flatten = list(uncertainties_windows.flatten())
    crash_window = uncertainties_windows_flatten[start_frame:current_frame]

    tot_window_TP, tot_window_FN = get_window_positive_negative(crash_window, threshold)

    window_TP, window_FN = label_crash_window(tot_window_TP, tot_window_FN)

    return {
        "window": list(crash_window),
        "start_frame": int(start_frame),
        "end_frame": int(current_frame),
        "is_crash": int(1),
        "TP": int(window_TP),
        "FN": int(window_FN),
        "FP": int(0),
        "TN": int(0),
    }


def get_nominal_window(i, window, threshold):
    assert len(window) == NORMAL_WINDOW_LENGTH

    window_crash = False
    window_FP, window_TN, tot_window_FP, tot_window_TN = 0, 0, 0, 0

    tot_window_FP, tot_window_TN = get_window_positive_negative(window, threshold)
    window_FP, window_TN = label_normal_window(tot_window_FP, tot_window_TN)

    return {
        "window": list(window),
        "start_frame": int(i * NORMAL_WINDOW_LENGTH),
        "end_frame": int((((i * NORMAL_WINDOW_LENGTH) + NORMAL_WINDOW_LENGTH)) - 1),
        "is_crash": int(window_crash),
        "FP": int(window_FP),
        "TN": int(window_TN),
        "TP": int(0),
        "FN": int(0),
    }


def get_crashes_frames_list(uncertainties_windows, crashes_per_frame):
    crashes_frames = []

    for i in range(len(uncertainties_windows)):
        for j in range(len(uncertainties_windows[i])):
            current_frame = (i * NORMAL_WINDOW_LENGTH) + j
            if (crashes_per_frame.get(current_frame) == 1) and (
                    crashes_per_frame.get(current_frame - 1) == 0
            ):
                crashes_frames.append(current_frame)

    return crashes_frames


def anomalous_win_analysis_alt(uncertainties_windows, crashes_per_frame, threshold):
    (
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    ) = (0, 0, 0, 0, 0)

    windows = []

    crashes_frames = get_crashes_frames_list(uncertainties_windows, crashes_per_frame)

    for frame in crashes_frames:
        window_before_crash = get_window_before_crash(
            uncertainties_windows, frame, threshold
        )

        windows.append(window_before_crash)

        crash_window = get_crash_window(uncertainties_windows, frame, threshold)

        windows.append(crash_window)

    # pprint(windows)
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

    assert (tot_windows_TP + tot_windows_FN + tot_windows_FP + tot_windows_TN) == len(
        windows
    )

    return (
        windows,
        tot_windows_TP,
        tot_windows_FN,
        tot_windows_FP,
        tot_windows_TN,
        tot_crashes,
    )


def anomalous_win_analysis(uncertainties_windows, crashes_per_frame, threshold):
    windows = []
    tot_windows_TP, tot_windows_FN, tot_crashes = (0, 0, 0)

    crashes_frames = get_crashes_frames_list(uncertainties_windows, crashes_per_frame)
    for frame in crashes_frames:
        crash_window = get_crash_window(uncertainties_windows, frame, threshold)

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


def nominal_win_analysis(uncertainties_windows, crashes_per_frame, threshold):
    windows = []

    for i in range(len(uncertainties_windows)):
        (window) = get_nominal_window(i, uncertainties_windows[i], threshold)
        windows.append(window)  # based on windows DB-schema columns

    return windows
