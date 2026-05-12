def handle_outliers(df):

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    for col in num_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    print("\nOutlier Treatment Completed")

    return df