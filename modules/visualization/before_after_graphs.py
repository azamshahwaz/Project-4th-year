import matplotlib.pyplot as plt
import seaborn as sns


def plot_before_after_counts(
    df_before,
    df_after,
    max_cols=4
):

    print("\n========== BEFORE vs AFTER COUNT PLOTS ==========")

    common_cols = [
        col for col in df_before.columns
        if col in df_after.columns
    ]

    selected_cols = common_cols[:max_cols]

    for col in selected_cols:

        if df_after[col].nunique() <= 10:

            plt.figure(figsize=(8, 4))

            # BEFORE
            plt.subplot(1, 2, 1)

            sns.countplot(
                x=df_before[col]
            )

            plt.title(f"Before: {col}")

            # AFTER
            plt.subplot(1, 2, 2)

            sns.countplot(
                x=df_after[col]
            )

            plt.title(f"After: {col}")

            plt.tight_layout()
            plt.show()