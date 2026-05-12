import numpy as np
import pandas as pd


# -----------------------------------
# Selection Bias
# -----------------------------------
def detect_selection_bias(df):

    for col in df.columns:

        if df[col].nunique() <= 10:

            if (
                df[col]
                .value_counts(normalize=True)
                .max()
                > 0.7
            ):
                return True

    return False


# -----------------------------------
# Sampling Bias
# -----------------------------------
def detect_sampling_bias(df):

    return df.shape[0] < 30


# -----------------------------------
# Response Bias
# -----------------------------------
def detect_response_bias(df):

    if df.shape[1] <= 1:
        return False

    corr = df.corr(numeric_only=True)

    np.fill_diagonal(corr.values, 0)

    return corr.abs().max().max() > 0.85


# -----------------------------------
# Label Bias
# -----------------------------------
def detect_label_bias(df):

    for col in df.columns:

        if df[col].nunique() <= 5:

            if (
                df[col]
                .value_counts(normalize=True)
                .max()
                > 0.75
            ):
                return True

    return False


# -----------------------------------
# Measurement Bias
# -----------------------------------
def detect_measurement_bias(df):

    num = df.select_dtypes(
        include=['int64', 'float64']
    )

    if len(num.columns) == 0:
        return False

    return num.std().max() > 500


# -----------------------------------
# Representation Bias
# -----------------------------------
def detect_representation_bias(df):

    for col in df.columns:

        if df[col].nunique() <= 10:

            if (
                df[col]
                .value_counts(normalize=True)
                .min()
                < 0.05
            ):
                return True

    return False


# -----------------------------------
# Proxy Bias
# -----------------------------------
def detect_proxy_bias(df):

    corr = df.corr(numeric_only=True)

    for i in range(len(corr.columns)):

        for j in range(i + 1, len(corr.columns)):

            if abs(corr.iloc[i, j]) > 0.9:
                return True

    return False


# -----------------------------------
# Confirmation Bias
# -----------------------------------
def detect_confirmation_bias(df):

    num = df.select_dtypes(
        include=['int64', 'float64']
    )

    if len(num.columns) == 0:
        return False

    return any(abs(num.skew()) > 1)


# -----------------------------------
# MAIN BIAS FUNCTION
# -----------------------------------
def detect_all_biases(df):

    print("\n========== BIAS DETECTION ==========")

    bias_report = {

        "Selection Bias":
            detect_selection_bias(df),

        "Sampling Bias":
            detect_sampling_bias(df),

        "Response Bias":
            detect_response_bias(df),

        "Label Bias":
            detect_label_bias(df),

        "Measurement Bias":
            detect_measurement_bias(df),

        "Representation Bias":
            detect_representation_bias(df),

        "Proxy Bias":
            detect_proxy_bias(df),

        "Confirmation Bias":
            detect_confirmation_bias(df)
    }

    return bias_report