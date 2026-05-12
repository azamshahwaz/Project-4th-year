# %%
# =========================================================
# RESPONSIBLE AI PIPELINE
# FINAL CLEAN UNIVERSAL VERSION
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# JUPYTER SUPPORT
# =========================================================

try:
    get_ipython().run_line_magic(
        'matplotlib',
        'inline'
    )
except:
    pass


# %%
# =========================================================
# IMPORT MODULES
# =========================================================

from modules.dataset.loader import load_dataset

from modules.dataset.validator import (
    validate_dataset
)

from modules.dataset.summary import (
    dataset_summary
)

from modules.preprocessing.preprocessing_pipeline import (
    preprocess_dataset
)

from modules.preprocessing.skewness_fixer import (
    fix_skewness
)

from modules.balancing.imbalance_detector import (
    detect_imbalance
)

from modules.balancing.imbalance_report import (
    imbalance_report
)

from modules.balancing.smote_module import (
    apply_smote
)

from modules.bias.bias_detector import (
    detect_all_biases
)

from modules.bias.bias_table import (
    generate_bias_table
)

from modules.bias.fairness_metrics import (
    calculate_fairness
)

from modules.bias.fairness_fix import (
    apply_fairness_fix
)

from modules.bias.proxy_bias_remover import (
    remove_proxy_bias
)

from modules.metrics.edqs import (
    calculate_edqs
)

from modules.metrics.eri import (
    calculate_eri
)

from modules.metrics.rai import (
    calculate_rai
)

from modules.explainability.llm_explainer import (
    explain_dataset
)

from modules.visualization.before_after_graphs import (
    plot_before_after_counts
)

from modules.visualization.boxplots import (
    plot_boxplots
)

from modules.visualization.heatmaps import (
    plot_heatmaps
)

from modules.visualization.piecharts import (
    plot_piecharts
)

from modules.visualization.edqs_graph import (
    plot_edqs_comparison
)

from modules.utils.logger import (
    set_log_file,
    write_log,
    log_section
)


# %%
# =========================================================
# UNIVERSAL FILE PICKER
# =========================================================

def choose_file():

    try:

        root = tk.Tk()

        root.withdraw()

        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(

            title="Select CSV Dataset",

            filetypes=[("CSV Files", "*.csv")]
        )

        root.destroy()

        if file_path:

            print("\nDataset Selected Successfully")

            return file_path

    except Exception as e:

        print("\nTkinter File Picker Failed")

        print(f"Reason: {e}")

    # =====================================================
    # FALLBACK INPUT
    # =====================================================

    print("\nEnter CSV File Path Manually")

    file_path = input("CSV Path: ").strip()

    file_path = file_path.replace('"', '')

    if os.path.exists(file_path):

        print("\nDataset Path Accepted")

        return file_path

    else:

        raise FileNotFoundError(
            f"\nFile Not Found:\n{file_path}"
        )


# %%
# =========================================================
# AUTO DETECT CATEGORICAL COLUMNS
# =========================================================

def detect_categorical_columns(
    df,
    target_col
):

    categorical_cols = []

    for col in df.columns:

        if col == target_col:
            continue

        unique_count = df[col].nunique()

        if df[col].dtype == "object":

            categorical_cols.append(col)

        elif unique_count <= 10:

            categorical_cols.append(col)

    return categorical_cols


# %%
# =========================================================
# TARGET COLUMN DETECTION
# =========================================================

def detect_target_column(df):

    possible_targets = [

        col for col in df.columns

        if any(

            keyword in col.lower()

            for keyword in [

                "target",
                "label",
                "class",
                "output",
                "result",
                "stroke"
            ]
        )
    ]

    if possible_targets:

        return possible_targets[-1]

    return df.columns[-1]


# %%
# =========================================================
# PRINT SECTION
# =========================================================

def print_section(title):

    print(f"\n{'=' * 15} {title} {'=' * 15}")


# %%
# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print_section(
        "RESPONSIBLE AI PIPELINE STARTED"
    )

    # =====================================================
    # FILE SELECTION
    # =====================================================

    file_path = choose_file()

    if not file_path:

        print("No Dataset Selected")

        return

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = load_dataset(file_path)

    # =====================================================
    # LOGGER
    # =====================================================

    dataset_name = os.path.basename(file_path)

    set_log_file(dataset_name)

    log_section("DATASET INFORMATION")

    write_log(f"Dataset Name: {dataset_name}")

    write_log(f"Dataset Shape: {df.shape}")

    write_log(f"Columns: {df.columns.tolist()}")

    # =====================================================
    # VALIDATION
    # =====================================================

    print_section("DATASET VALIDATION")

    validate_dataset(df)

    # =====================================================
    # SUMMARY
    # =====================================================

    print_section("DATASET SUMMARY")

    dataset_summary(df)

    # =====================================================
    # RAW COPY
    # =====================================================

    df_before = df.copy()

    # =====================================================
    # PREPROCESSING
    # =====================================================

    print_section("PREPROCESSING")

    df, encoders = preprocess_dataset(df)

    print("\nPreprocessing Completed")

    # =====================================================
    # TARGET DETECTION
    # =====================================================

    target_col = detect_target_column(df)

    print(f"\nSelected Target Column: {target_col}")

    log_section("TARGET COLUMN")

    write_log(
        f"Selected Target Column: {target_col}"
    )

    # =====================================================
    # EDQS BEFORE
    # =====================================================

    print_section("EDQS BEFORE RECTIFICATION")

    edqs_before_metrics = calculate_edqs(
        df,
        target_col
    )

    edqs_before = (
        edqs_before_metrics["edqs"]
    )

    for key, value in edqs_before_metrics.items():

        print(f"{key} : {value}")

    # =====================================================
    # IMBALANCE DETECTION
    # =====================================================

    print_section("IMBALANCE DETECTION")

    imbalance_ratio, class_counts = detect_imbalance(
        df,
        target_col
    )

    # =====================================================
    # IMBALANCE REPORT
    # =====================================================

    imbalance_report(class_counts)

    # =====================================================
    # SMOTE
    # =====================================================

    print_section("SMOTE BALANCING")

    categorical_cols = detect_categorical_columns(
        df,
        target_col
    )

    print("\nDetected Categorical Columns:")

    print(categorical_cols)

    try:

        df = apply_smote(
            df,
            target_col,
            categorical_cols
        )

    except Exception as e:

        print("\nSMOTE Failed")

        print(e)

    # =====================================================
    # BIAS DETECTION
    # =====================================================

    print_section("BIAS DETECTION")

    bias_report = detect_all_biases(df)

    for bias, status in bias_report.items():

        print(
            f"{bias} : "
            f"{'Detected' if status else 'Absent'}"
        )

    # =====================================================
    # BIAS TABLE
    # =====================================================

    generate_bias_table(
        bias_report
    )

    # =====================================================
    # FAIRNESS ANALYSIS
    # =====================================================

    print_section("FAIRNESS ANALYSIS")

    fairness_results, fairness_score = (
        calculate_fairness(
            df,
            target_col
        )
    )

    # =====================================================
    # FAIRNESS FIX
    # =====================================================

    print_section("FAIRNESS FIX")

    try:

        df = apply_fairness_fix(
            df,
            fairness_results,
            target_col
        )

    except Exception as e:

        print("\nFairness Fix Failed")

        print(e)

    # =====================================================
    # PROXY BIAS REMOVAL
    # =====================================================

    print_section("PROXY BIAS REMOVAL")

    try:

        df = remove_proxy_bias(
            df,
            target_col,
            threshold=0.80
        )

    except Exception as e:

        print("\nProxy Bias Removal Failed")

        print(e)

    # =====================================================
    # SKEWNESS FIX
    # =====================================================

    print_section("SKEWNESS FIX")

    try:

        df = fix_skewness(
            df,
            target_col
        )

    except Exception as e:

        print("\nSkewness Fix Failed")

        print(e)

    # =====================================================
    # RECHECK BIAS
    # =====================================================

    print_section("BIAS RECHECK")

    final_bias_report = detect_all_biases(df)

    for bias, status in final_bias_report.items():

        print(
            f"{bias} : "
            f"{'Detected' if status else 'Absent'}"
        )

    # =====================================================
    # EDQS AFTER
    # =====================================================

    print_section("EDQS AFTER RECTIFICATION")

    edqs_after_metrics = calculate_edqs(
        df,
        target_col
    )

    edqs_after = (
        edqs_after_metrics["edqs"]
    )

    for key, value in edqs_after_metrics.items():

        print(f"{key} : {value}")

    # =====================================================
    # IMPROVEMENT
    # =====================================================

    improvement = (
        edqs_after - edqs_before
    )

    print(
        f"\nEDQS Improvement : "
        f"{improvement:.2f}%"
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    print_section("VISUALIZATION")

    try:

        plot_before_after_counts(
            df_before,
            df
        )

        plot_boxplots(
            df_before,
            df
        )

        plot_heatmaps(
            df_before,
            df
        )

        plot_piecharts(
            df_before,
            df,
            categorical_cols
        )

        plot_edqs_comparison(
            edqs_before,
            edqs_after
        )

        plt.show()

        print("\nVisualizations Completed")

    except Exception as e:

        print("\nVisualization Error")

        print(e)

    # =====================================================
    # ERI
    # =====================================================

    print_section("ETHICAL RISK INDEX")

    missing_pct = (

        df.isnull().sum().sum()

        /

        (df.shape[0] * df.shape[1])

    ) * 100

    counts = df[target_col].value_counts()

    imbalance_ratio = (
        counts.min()
        /
        counts.max()
    )

    imbalance_pct = (
        1 - imbalance_ratio
    ) * 100

    bias_count = sum(
        1 for v in final_bias_report.values()
        if v
    )

    total_biases = len(final_bias_report)

    eri, risk = calculate_eri(

        missing_pct,

        imbalance_pct,

        bias_count,

        total_biases
    )

    # =====================================================
    # LLM EXPLANATION
    # =====================================================

    print_section("LLM EXPLANATION")

    try:

        explain_dataset(
            df,
            target_col,
            edqs_after,
            eri,
            final_bias_report
        )

    except Exception as e:

        print("\nLLM Explanation Failed")

        print(e)

    # =====================================================
    # RAI
    # =====================================================

    print_section("RESPONSIBLE AI INDEX")

    rai, status = calculate_rai(

        edqs_after,

        eri,

        bias_count,

        total_biases
    )

    # =====================================================
    # FINAL VERDICT
    # =====================================================

    print_section("FINAL AI VERDICT")

    if rai >= 80:

        verdict = "FAIR AI SYSTEM"

    elif rai >= 60:

        verdict = "MODERATE AI SYSTEM"

    else:

        verdict = "HIGH RISK AI SYSTEM"

    print(f"\nFinal Verdict : {verdict}")

    # =====================================================
    # SAVE FINAL DATASET
    # =====================================================

    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join(

        "outputs",

        "final_processed_dataset.csv"
    )

    df.to_csv(

        output_path,

        index=False
    )

    print(
        f"\nFinal Dataset Saved:\n{output_path}"
    )

    # =====================================================
    # LOGGING
    # =====================================================

    log_section("FINAL RESULTS")

    write_log(
        f"EDQS Before : {edqs_before:.2f}"
    )

    write_log(
        f"EDQS After : {edqs_after:.2f}"
    )

    write_log(
        f"ERI Score : {eri:.2f}"
    )

    write_log(
        f"RAI Score : {rai:.2f}"
    )

    write_log(
        f"Final Verdict : {verdict}"
    )

    write_log(
        f"Final Dataset Shape : {df.shape}"
    )

    # =====================================================
    # COMPLETED
    # =====================================================

    print_section(
        "RESPONSIBLE AI PIPELINE COMPLETED"
    )


# %%
# =========================================================
# RUN PIPELINE
# =========================================================

if __name__ == "__main__":

    main()