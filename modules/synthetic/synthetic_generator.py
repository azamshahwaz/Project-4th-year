# =========================================================
# SYNTHETIC DATA GENERATOR
# RESPONSIBLE AI VERSION
# =========================================================

import pandas as pd
import numpy as np


# =========================================================
# SYNTHETIC DATA GENERATION FUNCTION
# =========================================================

def generate_synthetic_data(
    df,
    target_col=None,
    ratio=0.2,
    noise_factor=0.02,
    random_state=42
):

    print("\n========== SYNTHETIC DATA GENERATION ==========")

    # =====================================================
    # COPY DATAFRAME
    # =====================================================

    df = df.copy()

    # =====================================================
    # MARK ORIGINAL DATA
    # =====================================================

    df["data_type"] = "original"

    # =====================================================
    # NUMERICAL COLUMNS
    # =====================================================

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove target column from numerical columns
    if target_col in num_cols:
        num_cols.remove(target_col)

    # =====================================================
    # INITIALIZE SYNTHETIC DATAFRAME
    # =====================================================

    synthetic_df = pd.DataFrame()

    # =====================================================
    # CASE 1: TARGET COLUMN EXISTS
    # =====================================================

    if (
        target_col is not None
        and target_col in df.columns
        and df[target_col].nunique() > 1
    ):

        # Class counts
        counts = df[target_col].value_counts()

        print("\nClass Distribution:")
        print(counts)

        imbalance_ratio = counts.min() / counts.max()

        print(f"\nImbalance Ratio : {imbalance_ratio:.2f}")

        # =================================================
        # DATASET ALREADY BALANCED
        # =================================================

        if imbalance_ratio > 0.7:

            print("\nDataset already balanced")
            print("Synthetic generation skipped")

            return df

        synthetic_list = []

        # Majority class size
        max_count = counts.max()

        # =================================================
        # GENERATE SYNTHETIC DATA FOR MINORITY CLASSES
        # =================================================

        for cls, count in counts.items():

            subset = df[df[target_col] == cls]

            # Only oversample minority classes
            if count < max_count:

                needed = int((max_count - count) * ratio)

                # Ensure at least 1 sample
                needed = max(1, needed)

                print(f"\nGenerating {needed} samples for class: {cls}")

                # =========================================
                # RANDOM SAMPLING
                # =========================================

                synthetic_samples = subset.sample(
                    n=needed,
                    replace=True,
                    random_state=random_state
                ).copy()

                # =========================================
                # ADD GAUSSIAN NOISE
                # =========================================

                if len(num_cols) > 0:

                    try:

                        std_values = (
                            df[num_cols]
                            .std()
                            .replace(0, 1)
                        )

                        noise = np.random.normal(
                            loc=0,
                            scale=std_values.values * noise_factor,
                            size=synthetic_samples[num_cols].shape
                        )

                        synthetic_samples[num_cols] = (
                            synthetic_samples[num_cols].astype(float) + noise
                        )

                    except Exception as e:

                        print("\nNoise Addition Failed")
                        print(e)

                # =========================================
                # MARK SYNTHETIC
                # =========================================

                synthetic_samples["data_type"] = "synthetic"

                synthetic_list.append(synthetic_samples)

        # =================================================
        # CONCAT ALL SYNTHETIC DATA
        # =================================================

        if synthetic_list:

            synthetic_df = pd.concat(
                synthetic_list,
                ignore_index=True
            )

    # =====================================================
    # CASE 2: NO TARGET COLUMN
    # =====================================================

    else:

        sample_size = max(1, int(len(df) * ratio))

        synthetic_df = df.sample(
            n=sample_size,
            replace=True,
            random_state=random_state
        ).copy()

        # =================================================
        # ADD NOISE
        # =================================================

        if len(num_cols) > 0:

            std_values = (
                df[num_cols]
                .std()
                .replace(0, 1)
            )

            noise = np.random.normal(
                loc=0,
                scale=std_values.values * noise_factor,
                size=synthetic_df[num_cols].shape
            )

            synthetic_df[num_cols] = (
                synthetic_df[num_cols].astype(float) + noise
            )

        synthetic_df["data_type"] = "synthetic"

    # =====================================================
    # FINAL MERGE
    # =====================================================

    final_df = pd.concat(
        [df, synthetic_df],
        ignore_index=True
    )

    # =====================================================
    # RESET INDEX
    # =====================================================

    final_df = final_df.reset_index(drop=True)

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print("\nSynthetic Data Added")

    print(f"Original Shape  : {df.shape}")
    print(f"Synthetic Shape : {synthetic_df.shape}")
    print(f"Final Shape     : {final_df.shape}")

    # =====================================================
    # DATA TYPE DISTRIBUTION
    # =====================================================

    print("\nData Type Distribution:")

    print(
        final_df["data_type"]
        .value_counts()
    )

    # =====================================================
    # CLASS DISTRIBUTION AFTER GENERATION
    # =====================================================

    if target_col in final_df.columns:

        print("\nFinal Class Distribution:")

        print(
            final_df[target_col]
            .value_counts()
        )

    # =====================================================
    # RETURN FINAL DATAFRAME
    # =====================================================

    return final_df