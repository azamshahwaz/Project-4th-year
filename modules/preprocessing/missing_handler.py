def handle_missing_values(df):

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    # Numerical
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing Values Handled")

    return df