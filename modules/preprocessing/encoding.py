from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(df):

    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    encoders = {}

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    print("\nCategorical Encoding Completed")

    return df, encoders