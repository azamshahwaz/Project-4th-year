import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmaps(
    df_before,
    df_after
):

    print("\n========== CORRELATION HEATMAPS ==========")

    num_before = df_before.select_dtypes(
        include=['int64', 'float64']
    )

    num_after = df_after.select_dtypes(
        include=['int64', 'float64']
    )

    if (
        len(num_before.columns) > 1
        and len(num_after.columns) > 1
    ):

        plt.figure(figsize=(10, 4))

        # BEFORE
        plt.subplot(1, 2, 1)

        sns.heatmap(
            num_before.corr(),
            annot=False
        )

        plt.title("Before Correlation")

        # AFTER
        plt.subplot(1, 2, 2)

        sns.heatmap(
            num_after.corr(),
            annot=False
        )

        plt.title("After Correlation")

        plt.tight_layout()
        plt.show()