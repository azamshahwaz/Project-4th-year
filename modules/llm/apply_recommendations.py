from modules.balancing.smote_module import (
    apply_smote
)

from modules.bias.proxy_bias_remover import (
    remove_proxy_bias
)

from modules.preprocessing.skewness_fixer import (
    fix_skewness
)


def apply_llm_recommendations(

    df,

    recommendations,

    target_col,

    categorical_cols
):

    print(
        "\nApplying LLM Recommendations"
    )

    # =====================================
    # REMOVE DUPLICATES
    # =====================================

    if recommendations.get(
        "remove_duplicates"
    ):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        print(
            f"Duplicates Removed : "
            f"{before - after}"
        )

    # =====================================
    # FILL MISSING VALUES
    # =====================================

    if recommendations.get(
        "fill_missing_values"
    ):

        for col in df.columns:

            if (
                df[col].dtype == "object"
            ):

                df[col] = (
                    df[col]
                    .fillna(
                        df[col]
                        .mode()[0]
                    )
                )

            else:

                df[col] = (
                    df[col]
                    .fillna(
                        df[col]
                        .median()
                    )
                )

        print(
            "Missing Values Filled"
        )

    # =====================================
    # APPLY SMOTE
    # =====================================

    if recommendations.get(
        "apply_smote"
    ):

        df = apply_smote(

            df,

            target_col,

            categorical_cols
        )

        print(
            "SMOTE Applied"
        )

    # =====================================
    # REMOVE PROXY BIAS
    # =====================================

    if recommendations.get(
        "remove_proxy_bias"
    ):

        df = remove_proxy_bias(

            df,

            target_col
        )

        print(
            "Proxy Bias Removed"
        )

    # =====================================
    # FIX SKEWNESS
    # =====================================

    if recommendations.get(
        "fix_skewness"
    ):

        df = fix_skewness(

            df,

            target_col
        )

        print(
            "Skewness Fixed"
        )

    return df