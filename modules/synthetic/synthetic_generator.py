import pandas as pd
import numpy as np


def generate_synthetic_data(
    df,
    target_col,
    ratio=0.2
):

    print("\n========== SYNTHETIC DATA GENERATION ==========")

    df = df.copy()

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    num_cols = [
        col for col in num_cols
        if col != target_col
    ]

    # -----------------------------------
    # CHECK TARGET
    # -----------------------------------
    if (
        target_col in df.columns
        and df[target_col].nunique() > 1
    ):

        counts = (
            df[target_col]
            .value_counts()
        )

        imbalance_ratio = (
            counts.min() /
            counts.max()
        )

        # Already balanced
        if imbalance_ratio > 0.7:

            print(
                "\nDataset already balanced"
            )

            print(
                "Synthetic generation skipped"
            )

            return df

        synthetic_list = []

        for cls in counts.index:

            subset = df[
                df[target_col] == cls
            ]

            needed = int(
                len(subset) * ratio
            )

            synthetic_samples = subset.sample(
                n=needed,
                replace=True,
                random_state=42
            ).copy()

            # Add noise
            if num_cols:

                noise = np.random.normal(
                    loc=0,
                    scale=0.01,
                    size=synthetic_samples[
                        num_cols
                    ].shape
                )

                synthetic_samples[
                    num_cols
                ] += noise

            synthetic_list.append(
                synthetic_samples
            )

        synthetic_df = pd.concat(
            synthetic_list,
            ignore_index=True
        )

    else:

        synthetic_df = df.sample(
            frac=ratio,
            replace=True,
            random_state=42
        ).copy()

    # -----------------------------------
    # FINAL MERGE
    # -----------------------------------
    final_df = pd.concat(
        [df, synthetic_df],
        ignore_index=True
    )

    print("\nSynthetic Data Added")

    print(
        f"Original Shape : {df.shape}"
    )

    print(
        f"New Shape      : {final_df.shape}"
    )

    return final_df