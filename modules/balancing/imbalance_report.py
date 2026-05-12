def imbalance_report(class_counts):

    print("\n========== IMBALANCE REPORT ==========")

    total = class_counts.sum()

    for cls, count in class_counts.items():

        percent = (count / total) * 100

        print(
            f"Class {cls} : "
            f"{count} samples "
            f"({percent:.2f}%)"
        )