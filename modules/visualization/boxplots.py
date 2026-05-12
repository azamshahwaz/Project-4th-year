import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplots(
    df_before,
    df_after,
    max_cols=4
):

    print("\n========== BEFORE vs AFTER BOXPLOTS ==========")

    num_before = df_before.select_dtypes(
        include=['int64', 'float64']
    ).columns

    num_after = df_after.select_dtypes(
        include=['int64', 'float64']
    ).columns

    valid_cols = [
        col for col in num_before
        if col in num_after
    ]

    for col in valid_cols[:max_cols]:

        plt.figure(figsize=(8, 4))

        # BEFORE
        plt.subplot(1, 2, 1)

        sns.boxplot(
            y=df_before[col]
        )

        plt.title(f"Before: {col}")

        # AFTER
        plt.subplot(1, 2, 2)

        sns.boxplot(
            y=df_after[col]
        )

        plt.title(f"After: {col}")

        plt.tight_layout()
        plt.show()