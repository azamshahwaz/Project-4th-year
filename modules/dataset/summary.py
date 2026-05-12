def dataset_summary(df):

    print("\nDATASET SUMMARY")
    print("-" * 40)

    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())