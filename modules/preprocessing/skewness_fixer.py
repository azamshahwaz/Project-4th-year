import numpy as np


def fix_skewness(
    df,
    target_col
):

    print("\n========== SKEWNESS FIX STARTED ==========")

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    for col in num_cols:

        if (
            col in df.columns
            and col != target_col
        ):

            try:

                skew_val = df[col].skew()

                # -----------------------------------
                # HIGH SKEWNESS
                # -----------------------------------
                if abs(skew_val) > 1:

                    # Positive only
                    if (df[col] > 0).all():

                        df[col] = np.log1p(
                            df[col]
                        )

                        print(
                            f"{col} → "
                            f"Log Transform"
                        )

                    # Contains negatives
                    else:

                        shift = (
                            abs(df[col].min())
                            + 1
                        )

                        df[col] = np.log1p(
                            df[col] + shift
                        )

                        print(
                            f"{col} → "
                            f"Shift + Log"
                        )

                else:

                    print(
                        f"{col} → "
                        f"No Fix Needed"
                    )

            except Exception as e:

                print(
                    f"{col} → Error: {e}"
                )

    print("\n========== SKEWNESS FIX COMPLETED ==========")

    return df