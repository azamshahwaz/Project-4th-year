from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


# =========================================================
# APPLY SMOTE USING SMOTENC
# =========================================================

def apply_smote(
    df,
    target_col,
    categorical_cols
):

    print("\n========== SMOTE STARTED ==========")

    # =====================================================
    # SAFETY COPY
    # =====================================================

    df = df.copy()

    # =====================================================
    # FEATURES & TARGET
    # =====================================================

    X = df.drop(columns=[target_col])

    y = df[target_col]

    # =====================================================
    # REMOVE NULLS IF ANY
    # =====================================================

    full_df = pd.concat([X, y], axis=1)

    full_df = full_df.dropna()

    X = full_df.drop(columns=[target_col])

    y = full_df[target_col]

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    class_counts = y.value_counts()

    imbalance_ratio = (
        class_counts.min()
        /
        class_counts.max()
    )

    print("\nBefore SMOTE:")

    print(class_counts)

    print(
        f"\nImbalance Ratio : "
        f"{imbalance_ratio:.4f}"
    )

    # =====================================================
    # DATASET ALREADY BALANCED
    # =====================================================

    if imbalance_ratio >= 0.95:

        print("\nDataset Already Balanced")

        print("\n========== SMOTE SKIPPED ==========")

        return df

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # =====================================================
    # CATEGORICAL COLUMN INDICES
    # =====================================================

    categorical_indices = []

    for col in categorical_cols:

        if col in X.columns:

            categorical_indices.append(
                X.columns.get_loc(col)
            )

    print("\nCategorical Columns:")

    print(categorical_cols)

    print("\nCategorical Indices:")

    print(categorical_indices)

    # =====================================================
    # SAFE K NEIGHBORS
    # =====================================================

    minority_count = y_train.value_counts().min()

    if minority_count <= 1:

        k_neighbors = 1

    else:

        k_neighbors = min(
            5,
            minority_count - 1
        )

    print(f"\nk_neighbors Used : {k_neighbors}")

    # =====================================================
    # STORE BEFORE COUNTS
    # =====================================================

    before_counts = y_train.value_counts()

    # =====================================================
    # APPLY SMOTENC
    # =====================================================

    try:

        smote = SMOTENC(

            categorical_features=categorical_indices,

            random_state=42,

            k_neighbors=k_neighbors
        )

        X_resampled, y_resampled = smote.fit_resample(

            X_train,

            y_train
        )

        print("\nSMOTENC Applied Successfully")

    except Exception as e:

        print("\nSMOTENC FAILED")

        print(e)

        return df

    # =====================================================
    # AFTER COUNTS
    # =====================================================

    after_counts = pd.Series(
        y_resampled
    ).value_counts()

    print("\nBefore SMOTE (TRAIN):")

    print(before_counts)

    print("\nAfter SMOTE (TRAIN):")

    print(after_counts)

    print("\nTest Data Distribution:")

    print(
        pd.Series(y_test).value_counts()
    )

    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    X_resampled = pd.DataFrame(

        X_resampled,

        columns=X.columns
    )

    y_resampled = pd.Series(

        y_resampled,

        name=target_col
    )

    # =====================================================
    # FIX CATEGORICAL COLUMNS
    # =====================================================

    for col in categorical_cols:

        if col in X_resampled.columns:

            # Round categorical values
            X_resampled[col] = (
                X_resampled[col]
                .round()
                .astype(int)
            )

    # =====================================================
    # FIX TARGET TYPE
    # =====================================================

    y_resampled = y_resampled.astype(int)

    # =====================================================
    # TRAIN DATAFRAME
    # =====================================================

    train_df = pd.concat(

        [
            X_resampled,
            y_resampled
        ],

        axis=1
    )

    # =====================================================
    # TEST DATAFRAME
    # =====================================================

    test_df = pd.concat(

        [
            pd.DataFrame(
                X_test,
                columns=X.columns
            ).reset_index(drop=True),

            pd.Series(
                y_test,
                name=target_col
            ).reset_index(drop=True)
        ],

        axis=1
    )

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    final_df = pd.concat(

        [
            train_df,
            test_df
        ],

        ignore_index=True
    )

    # =====================================================
    # RESET INDEX
    # =====================================================

    final_df.reset_index(
        drop=True,
        inplace=True
    )

    # =====================================================
    # FINAL DISTRIBUTION
    # =====================================================

    final_counts = (
        final_df[target_col]
        .value_counts()
    )

    final_ratio = (
        final_counts.min()
        /
        final_counts.max()
    )

    print("\nFinal Combined Distribution:")

    print(final_counts)

    print(
        f"\nFinal Imbalance Ratio : "
        f"{final_ratio:.4f}"
    )

    print(
        f"\nFinal Shape : "
        f"{final_df.shape}"
    )

    # =====================================================
    # VERIFY CATEGORICALS
    # =====================================================

    print("\nCategorical Column Verification:")

    for col in categorical_cols:

        if col in final_df.columns:

            unique_values = sorted(
                final_df[col]
                .dropna()
                .unique()
                .tolist()
            )

            print(
                f"{col} -> "
                f"{unique_values[:15]}"
            )

    print("\n========== SMOTE COMPLETED ==========")

    return final_df