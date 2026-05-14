# =========================================================
# CTGAN SYNTHETIC DATA GENERATOR
# =========================================================

import pandas as pd

from ctgan import CTGAN

import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

def generate_ctgan_data(
    df,
    target_col,
    epochs=100
):

    print("\nTraining CTGAN Model...")

    # =====================================================
    # DETECT CATEGORICAL COLUMNS
    # =====================================================

    categorical_columns = [

        col for col in df.columns

        if df[col].dtype == "object"

        or df[col].nunique() <= 10
    ]

    # =====================================================
    # INITIALIZE MODEL
    # =====================================================

    model = CTGAN(

        epochs=epochs,

        verbose=True
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(

        df,

        categorical_columns
    )

    print("\nCTGAN Training Completed")

    # =====================================================
    # GENERATE SYNTHETIC SAMPLES
    # =====================================================

    synthetic_count = int(
        len(df) * 0.30
    )

    synthetic_df = model.sample(
        synthetic_count
    )

    print(
        f"\nSynthetic Samples Generated : "
        f"{len(synthetic_df)}"
    )

    # =====================================================
    # FIX NEGATIVE VALUES
    # =====================================================

    if "Timestamp" in synthetic_df.columns:

        synthetic_df["Timestamp"] = (

            synthetic_df["Timestamp"]

            .clip(lower=0)
        )

    # =====================================================
    # FIX INTEGER COLUMNS
    # =====================================================

    integer_columns = [

        "Timestamp",

        "Activities",

        "DHBreakfast",

        "NDHBreakfast",

        "DHBoxes",

        "NDHBoxes",

        "NDHMeals",

        "Nutrition"
    ]

    for col in integer_columns:

        if col in synthetic_df.columns:

            synthetic_df[col] = (

                synthetic_df[col]

                .round()

                .astype(int)
            )

    # =====================================================
    # FIX CATEGORICAL COLUMNS
    # =====================================================

    categorical_constraints = {

        "Gender": [0, 1],

        "Boarding": [0, 1],

        "Athlete": [0, 1],

        "BClass": [0, 1]
    }

    for col, allowed_values in (

        categorical_constraints.items()
    ):

        if col in synthetic_df.columns:

            synthetic_df[col] = (

                synthetic_df[col]

                .round()

                .astype(int)
            )

            synthetic_df[col] = (

                synthetic_df[col]

                .apply(

                    lambda x:

                    x

                    if x in allowed_values

                    else allowed_values[0]
                )
            )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    synthetic_df = (
        synthetic_df
        .drop_duplicates()
    )

    # =====================================================
    # RESET INDEX
    # =====================================================

    synthetic_df = (
        synthetic_df
        .reset_index(drop=True)
    )

    print(
        "\nSynthetic Data Cleaned Successfully"
    )

    return synthetic_df