def calculate_eri(
    missing_pct,
    imbalance_pct,
    bias_count,
    total_biases
):

    print("\n========== ERI CALCULATION ==========")

    missing_norm = missing_pct / 100

    imbalance_norm = imbalance_pct / 100

    bias_norm = (
        bias_count / total_biases
    )

    # Weighted risk model
    w1, w2, w3 = 0.4, 0.4, 0.2

    eri = (

        (w1 * missing_norm)

        +

        (w2 * imbalance_norm)

        +

        (w3 * bias_norm)

    ) * 100

    # -----------------------------------
    # RISK LEVEL
    # -----------------------------------
    if eri < 30:

        risk = "LOW RISK"

    elif eri < 60:

        risk = "MODERATE RISK"

    else:

        risk = "HIGH RISK"

    print(f"\nERI Score : {eri:.2f}")

    print(f"Risk Level: {risk}")

    return eri, risk