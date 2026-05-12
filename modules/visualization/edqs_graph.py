import matplotlib.pyplot as plt


def plot_edqs_comparison(
    before_score,
    after_score
):

    print("\n========== EDQS COMPARISON ==========")

    labels = [
        "EDQS Before",
        "EDQS After"
    ]

    values = [
        before_score,
        after_score
    ]

    plt.figure(figsize=(6, 5))

    bars = plt.bar(
        labels,
        values
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    plt.ylim(0, 100)

    plt.ylabel(
        "Ethical Data Quality Score (%)"
    )

    plt.title(
        "EDQS Before vs After"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.6
    )

    plt.show()