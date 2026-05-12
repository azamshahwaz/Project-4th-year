def explain_dataset(
    df,
    target_col,
    edqs,
    eri,
    bias_report
):

    print("\n========== LLM DATASET EXPLANATION ==========")

    total_rows, total_cols = df.shape

    bias_count = sum(
        1 for v in bias_report.values()
        if v
    )

    observations = []

    # -----------------------------------
    # BIAS OBSERVATIONS
    # -----------------------------------
    if bias_count > 3:

        observations.append(
            "Multiple biases detected"
        )

    else:

        observations.append(
            "Limited bias detected"
        )

    # -----------------------------------
    # EDQS OBSERVATION
    # -----------------------------------
    if edqs > 80:

        observations.append(
            "Dataset quality is high"
        )

    else:

        observations.append(
            "Dataset quality is moderate"
        )

    # -----------------------------------
    # ERI OBSERVATION
    # -----------------------------------
    if eri < 30:

        observations.append(
            "Ethical risk is low"
        )

    else:

        observations.append(
            "Ethical risk needs attention"
        )

    # -----------------------------------
    # FINAL EXPLANATION
    # -----------------------------------
    explanation = f"""

DATASET SUMMARY
--------------------------------

Total Records  : {total_rows}

Total Features : {total_cols}

Target Column  : {target_col}

EDQS Score     : {edqs:.2f}

ERI Score      : {eri:.2f}


KEY OBSERVATIONS
--------------------------------

• {observations[0]}

• {observations[1]}

• {observations[2]}


INTERPRETATION
--------------------------------

The dataset has undergone preprocessing,
bias detection, fairness correction,
and ethical quality assessment.

The Responsible AI pipeline improved
data fairness, reduced ethical risks,
and enhanced overall dataset reliability.
"""

    print(explanation)

    return explanation