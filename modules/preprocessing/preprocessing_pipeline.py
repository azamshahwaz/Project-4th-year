from modules.preprocessing.cleaning import (
    clean_column_names,
    remove_duplicates
)

from modules.preprocessing.missing_handler import (
    handle_missing_values
)

from modules.preprocessing.outlier_handler import (
    handle_outliers
)

from modules.preprocessing.encoding import (
    encode_categorical_features
)


def preprocess_dataset(df):

    print("\n========== PREPROCESSING STARTED ==========")

    # Column cleaning
    df = clean_column_names(df)

    # Duplicate removal
    df = remove_duplicates(df)

    # Missing values
    df = handle_missing_values(df)

    # Outlier handling
    df = handle_outliers(df)

    # Encoding
    df, encoders = encode_categorical_features(df)

    print("\n========== PREPROCESSING COMPLETED ==========")

    return df, encoders