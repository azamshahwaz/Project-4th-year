import numpy as np


def remove_proxy_bias(
    df,
    target_col,
    threshold=0.80
):

    print("\n========== PROXY BIAS REMOVAL ==========")

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    drop_cols = []

    if len(num_cols) > 1:

        corr_matrix = (
            df[num_cols]
            .corr()
            .abs()
        )

        upper_triangle = corr_matrix.where(

            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        for col in upper_triangle.columns:

            high_corr = upper_triangle[col][
                upper_triangle[col] > threshold
            ]

            if len(high_corr) > 0:

                if col == target_col:
                    continue

                correlated_cols = (
                    list(high_corr.index)
                    + [col]
                )

                variances = (
                    df[correlated_cols]
                    .var()
                )

                drop_feature = (
                    variances.idxmin()
                )

                if drop_feature != target_col:

                    drop_cols.append(
                        drop_feature
                    )

    drop_cols = list(set(drop_cols))

    df.drop(
        columns=drop_cols,
        inplace=True,
        errors='ignore'
    )

    print("\nRemoved Columns:")
    print(drop_cols if drop_cols else "None")

    print(
        f"\nRemaining Features: "
        f"{len(df.columns)}"
    )

    return df