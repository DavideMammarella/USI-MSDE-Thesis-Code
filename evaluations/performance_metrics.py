def _calculate_all(
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


def main():
    print("This script is not intended to be run directly.")


if __name__ == "__main__":
    main()
