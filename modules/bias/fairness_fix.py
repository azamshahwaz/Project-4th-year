import pandas as pd


def apply_fairness_fix(
    df,
    fairness_results,
    target_col
):

    print("\n========== FAIRNESS FIX STARTED ==========")

    high_bias_cols = []

    # -----------------------------------
    # FIND HIGH BIAS COLUMNS
    # -----------------------------------
    for col, val in fairness_results.items():

        if val["gap"] > 0.25:

            high_bias_cols.append(col)

    print("\nHigh Bias Columns:")
    print(high_bias_cols)

    # -----------------------------------
    # APPLY FIX
    # -----------------------------------
    for col in high_bias_cols:

        if (
            col in df.columns
            and col != target_col
        ):

            try:

                unique_vals = df[col].nunique()

                # -------------------------
                # Numerical-like
                # -------------------------
                if unique_vals > 5:

                    df[col] = pd.qcut(
                        df[col],
                        q=3,
                        labels=False,
                        duplicates='drop'
                    )

                    print(
                        f"{col} → "
                        f"Binning Applied"
                    )

                # -------------------------
                # Low diversity
                # -------------------------
                elif unique_vals <= 3:

                    df.drop(
                        columns=[col],
                        inplace=True
                    )

                    print(
                        f"{col} → "
                        f"Dropped"
                    )

                else:

                    print(
                        f"{col} → "
                        f"Kept"
                    )

            except Exception as e:

                print(
                    f"{col} → Error: {e}"
                )

    print("\n========== FAIRNESS FIX COMPLETED ==========")

    return df