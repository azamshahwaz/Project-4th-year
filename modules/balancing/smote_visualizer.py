import matplotlib.pyplot as plt
import seaborn as sns


def visualize_smote(before_counts, after_counts):

    plt.figure(figsize=(10, 4))

    # Before
    plt.subplot(1, 2, 1)

    sns.barplot(
        x=before_counts.index.astype(str),
        y=before_counts.values
    )

    plt.title("Before SMOTE")

    # After
    plt.subplot(1, 2, 2)

    sns.barplot(
        x=after_counts.index.astype(str),
        y=after_counts.values
    )

    plt.title("After SMOTE")

    plt.tight_layout()
    plt.show()