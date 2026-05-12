def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    )

    print("\nColumn Cleaning Completed")

    return df


def remove_duplicates(df):

    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    print(f"\nDuplicates Removed: {before - after}")

    return df