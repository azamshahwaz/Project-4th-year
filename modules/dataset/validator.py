def validate_dataset(df):

    print("\nRunning Dataset Validation...")

    # Empty check
    if df.empty:
        raise ValueError("Dataset is empty")

    # Duplicate check
    duplicates = df.duplicated().sum()

    # Missing values
    missing = df.isnull().sum().sum()

    print(f"Duplicate Rows : {duplicates}")
    print(f"Missing Values : {missing}")

    print("Validation Completed Successfully")

    return True