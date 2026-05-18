# =========================================================
# UNIVERSAL RESPONSIBLE AI BIAS DETECTOR
# =========================================================

import pandas as pd
import numpy as np

# =========================================================
# PROBABILITY NORMALIZER
# =========================================================

def normalize_probability(value):

    value = abs(value)

    value = min(max(value, 0), 1)

    return round(value, 2)

# =========================================================
# BIAS STATUS
# =========================================================

def bias_status(probability):

    if probability >= 0.6:

        return "High Bias"

    elif probability >= 0.4:

        return "Moderate Bias"

    else:

        return "Low Bias"

# =========================================================
# MAIN FAIRNESS FUNCTION
# =========================================================

def calculate_fairness(
    df,
    target_col
):

    print(
        "\n========== UNIVERSAL BIAS ANALYSIS =========="
    )

    bias_results = []

    total_rows = len(df)

    # =====================================================
    # 1. SELECTION BIAS
    # =====================================================

    class_distribution = (
        df[target_col]
        .value_counts(normalize=True)
    )

    selection_bias = (
        class_distribution.max()
        -
        class_distribution.min()
    )

    selection_bias = normalize_probability(
        selection_bias
    )

    bias_results.append({

        "Bias Type":
        "Selection Bias",

        "Probability":
        selection_bias,

        "Bias Status":
        bias_status(selection_bias)
    })

    # =====================================================
    # 2. SAMPLING BIAS
    # =====================================================

    sampling_bias = abs(

        len(df.sample(frac=0.5))

        / total_rows

        - 0.5
    )

    sampling_bias = normalize_probability(
        sampling_bias
    )

    bias_results.append({

        "Bias Type":
        "Sampling Bias",

        "Probability":
        sampling_bias,

        "Bias Status":
        bias_status(sampling_bias)
    })

    # =====================================================
    # 3. RESPONSE BIAS
    # =====================================================

    missing_ratio = (

        df.isnull()
        .sum()
        .sum()

        /

        (df.shape[0] * df.shape[1])
    )

    response_bias = normalize_probability(
        missing_ratio
    )

    bias_results.append({

        "Bias Type":
        "Response Bias",

        "Probability":
        response_bias,

        "Bias Status":
        bias_status(response_bias)
    })

    # =====================================================
    # 4. LABEL BIAS
    # =====================================================

    label_bias = (

        class_distribution.std()
    )

    label_bias = normalize_probability(
        label_bias
    )

    bias_results.append({

        "Bias Type":
        "Label Bias",

        "Probability":
        label_bias,

        "Bias Status":
        bias_status(label_bias)
    })

    # =====================================================
    # 5. MEASUREMENT BIAS
    # =====================================================

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    measurement_bias = 0

    if len(numeric_cols) > 0:

        measurement_bias = np.mean([

            abs(df[col].skew())

            for col in numeric_cols

        ]) / 10

    measurement_bias = normalize_probability(
        measurement_bias
    )

    bias_results.append({

        "Bias Type":
        "Measurement Bias",

        "Probability":
        measurement_bias,

        "Bias Status":
        bias_status(measurement_bias)
    })

    # =====================================================
    # 6. REPRESENTATION BIAS
    # =====================================================

    representation_bias = (

        1
        -
        class_distribution.min()
    )

    representation_bias = normalize_probability(
        representation_bias
    )

    bias_results.append({

        "Bias Type":
        "Representation Bias",

        "Probability":
        representation_bias,

        "Bias Status":
        bias_status(representation_bias)
    })

    # =====================================================
    # 7. PROXY BIAS
    # =====================================================

    proxy_bias = 0

    if len(numeric_cols) > 1:

        corr_matrix = (
            df[numeric_cols]
            .corr()
            .abs()
        )

        proxy_bias = (
            corr_matrix.mean().mean()
        ) / 2

    proxy_bias = normalize_probability(
        proxy_bias
    )

    bias_results.append({

        "Bias Type":
        "Proxy Bias",

        "Probability":
        proxy_bias,

        "Bias Status":
        bias_status(proxy_bias)
    })

    # =====================================================
    # 8. CONFIRMATION BIAS
    # =====================================================

    confirmation_bias = 0

    if len(numeric_cols) > 0:

        variances = [

            df[col].var()

            for col in numeric_cols
        ]

        confirmation_bias = (

            1
            /
            (
                np.mean(variances)
                + 1
            )
        )

    confirmation_bias = normalize_probability(
        confirmation_bias
    )

    bias_results.append({

        "Bias Type":
        "Confirmation Bias",

        "Probability":
        confirmation_bias,

        "Bias Status":
        bias_status(confirmation_bias)
    })

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    bias_df = pd.DataFrame(
        bias_results
    )

    print(
        "\n========== BIAS REPORT ==========\n"
    )

    print(
        bias_df.to_string(index=False)
    )

    # =====================================================
    # FAIRNESS SCORE
    # =====================================================

    avg_bias = bias_df[
        "Probability"
    ].mean()

    fairness_score = round(
        1 - avg_bias,
        2
    )

    return bias_df, fairness_score