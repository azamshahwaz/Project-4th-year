import pandas as pd


def generate_bias_table(bias_report):

    bias_df = pd.DataFrame({

        "Bias Type": bias_report.keys(),

        "Detected": [
            "Yes" if v else "No"
            for v in bias_report.values()
        ]
    })

    print("\n========== BIAS DETECTION TABLE ==========")

    print(bias_df)

    return bias_df