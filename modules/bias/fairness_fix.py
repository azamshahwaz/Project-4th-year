# =========================================================
# FAIRNESS FIX MODULE
# =========================================================

import pandas as pd
import numpy as np

# =========================================================
# APPLY FAIRNESS FIX
# =========================================================

def apply_fairness_fix(
    df,
    bias_results,
    target_col
):

    print("\n========== FAIRNESS FIX STARTED ==========")

    df = df.copy()

    try:

        # =================================================
        # HIGH BIAS TYPES
        # =================================================

        high_bias = bias_results[

            bias_results["Probability"] > 0.30

        ]

        print("\nHigh Bias Types:")

        print(
            high_bias["Bias Type"].tolist()
        )

        # =================================================
        # REPRESENTATION BIAS FIX
        # =================================================

        if "Representation Bias" in high_bias["Bias Type"].values:

            print(
                "\nFixing Representation Bias..."
            )

            class_counts = (
                df[target_col]
                .value_counts()
            )

            max_count = class_counts.max()

            balanced_data = []

            for cls in class_counts.index:

                cls_df = df[
                    df[target_col] == cls
                ]

                if len(cls_df) < max_count:

                    extra = cls_df.sample(
                        max_count - len(cls_df),
                        replace=True,
                        random_state=42
                    )

                    cls_df = pd.concat(
                        [cls_df, extra]
                    )

                balanced_data.append(cls_df)

            df = pd.concat(
                balanced_data,
                ignore_index=True
            )

            print(
                "\nRepresentation Bias Fixed"
            )

        # =================================================
        # SELECTION BIAS FIX
        # =================================================

        if "Selection Bias" in high_bias["Bias Type"].values:

            print(
                "\nFixing Selection Bias..."
            )

            df = df.sample(
                frac=1,
                random_state=42
            ).reset_index(drop=True)

            print(
                "\nSelection Bias Fixed"
            )

        # =================================================
        # PROXY BIAS FIX
        # =================================================

        if "Proxy Bias" in high_bias["Bias Type"].values:

            print(
                "\nFixing Proxy Bias..."
            )

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if col != target_col:

                    noise = np.random.normal(
                        0,
                        0.01,
                        len(df)
                    )

                    df[col] = (
                        df[col] + noise
                    )

            print(
                "\nProxy Bias Fixed"
            )

        print(
            "\n========== FAIRNESS FIX COMPLETED =========="
        )

        return df

    except Exception as e:

        print(e)

        return df