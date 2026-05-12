import pandas as pd


def detect_imbalance(df, target_col):

    print("\n========== IMBALANCE DETECTION ==========")

    class_counts = df[target_col].value_counts()

    print("\nClass Distribution:")
    print(class_counts)

    if len(class_counts) > 1:
        imbalance_ratio = (
            class_counts.min() /
            class_counts.max()
        )
    else:
        imbalance_ratio = 1

    imbalance_percent = (
        1 - imbalance_ratio
    ) * 100

    print(f"\nImbalance Ratio : {imbalance_ratio:.2f}")
    print(f"Imbalance %     : {imbalance_percent:.2f}%")

    return imbalance_ratio, class_counts