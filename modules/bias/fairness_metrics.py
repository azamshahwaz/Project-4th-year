import numpy as np


def demographic_parity(
    df,
    target_col,
    group_col
):

    rates = (
        df.groupby(group_col)[target_col]
        .mean()
    )

    gap = rates.max() - rates.min()

    return rates, gap


def calculate_fairness(
    df,
    target_col
):

    print("\n========== FAIRNESS ANALYSIS ==========")

    fairness_results = {}

    for col in df.columns:

        if (
            col != target_col
            and df[col].nunique() <= 10
        ):

            try:

                rates, gap = demographic_parity(
                    df,
                    target_col,
                    col
                )

                fairness_results[col] = {

                    "rates": rates,
                    "gap": gap
                }

            except:
                pass

    # -----------------------------------
    # OUTPUT
    # -----------------------------------
    for col, val in fairness_results.items():

        print(f"\nColumn: {col}")

        print("\nGroup Rates:")
        print(val["rates"])

        print(f"\nFairness Gap: {val['gap']:.3f}")

    # -----------------------------------
    # OVERALL SCORE
    # -----------------------------------
    if fairness_results:

        avg_gap = np.mean(
            [
                v["gap"]
                for v in fairness_results.values()
            ]
        )

        fairness_score = (
            1 - avg_gap
        ) * 100

    else:

        fairness_score = 100

    print(
        f"\nOverall Fairness Score: "
        f"{fairness_score:.2f}%"
    )

    return fairness_results, fairness_score