import numpy as np


def calculate_edqs(
    df,
    target_col=None
):

    # =====================================================
    # BASIC INFO
    # =====================================================
    total_rows, total_cols = df.shape

    total_values = total_rows * total_cols

    # =====================================================
    # MISSING VALUES
    # =====================================================
    missing_values = df.isnull().sum().sum()

    missing_ratio = (
        missing_values / total_values
        if total_values > 0
        else 0
    )

    missing_pct = missing_ratio * 100

    # =====================================================
    # DUPLICATE ROWS
    # =====================================================
    duplicate_rows = df.duplicated().sum()

    duplicate_ratio = (
        duplicate_rows / total_rows
        if total_rows > 0
        else 0
    )

    duplicate_pct = duplicate_ratio * 100

    # =====================================================
    # TARGET IMBALANCE
    # =====================================================
    if (
        target_col
        and target_col in df.columns
    ):

        counts = df[target_col].value_counts()

        if len(counts) > 1:

            imbalance_ratio = (
                counts.min()
                /
                counts.max()
            )

        else:

            imbalance_ratio = 1

    else:

        imbalance_ratio = 1

    imbalance_pct = (
        1 - imbalance_ratio
    ) * 100

    # =====================================================
    # EDQS WEIGHTS
    # =====================================================
    w_missing = 0.4
    w_duplicate = 0.3
    w_imbalance = 0.3

    # =====================================================
    # EDQS CALCULATION
    # =====================================================
    edqs = (

        w_missing * (1 - missing_ratio)

        +

        w_duplicate * (1 - duplicate_ratio)

        +

        w_imbalance * imbalance_ratio

    ) * 100

    # Safety Bound
    edqs = max(
        0,
        min(100, edqs)
    )

    # =====================================================
    # METRICS
    # =====================================================
    metrics = {

        "rows": total_rows,

        "cols": total_cols,

        "missing_values": missing_values,

        "duplicate_rows": duplicate_rows,

        "missing_pct": missing_pct,

        "duplicate_pct": duplicate_pct,

        "imbalance_ratio": imbalance_ratio,

        "imbalance_pct": imbalance_pct,

        "edqs": edqs
    }

    return metrics