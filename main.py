# =========================================================
# RESPONSIBLE AI PIPELINE
# FINAL FULLY FIXED OPTIMIZED VERSION
# BLANK GRAPH + MEMORY + CTGAN + FAST MODE FIXED
# =========================================================

import os
import gc
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# MATPLOTLIB BACKEND FIX
# =========================================================

import matplotlib
matplotlib.use("Agg")

# =========================================================
# FAST MODE
# =========================================================

FAST_MODE = True

# =========================================================
# BASIC LIBRARIES
# =========================================================

import tkinter as tk
from tkinter import filedialog

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

# =========================================================
# SKLEARN
# =========================================================

from sklearn.model_selection import train_test_split

# =========================================================
# FAIRLEARN
# =========================================================

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

# =========================================================
# REPORTLAB
# =========================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from xml.sax.saxutils import escape

# =========================================================
# PROJECT MODULES
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

from modules.synthetic.ctgan_generator import (
    generate_ctgan_data
)

from modules.bias.fairness_metrics import (
    calculate_fairness
)

from modules.bias.fairness_fix import (
    apply_fairness_fix
)

from modules.metrics.edqs import (
    calculate_edqs
)

from modules.model.train_model import (
    train_model
)

from modules.model.evaluate_model import (
    evaluate_model
)

from modules.model.save_model import (
    save_model
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
    write_log
)

# =========================================================
# MEMORY OPTIMIZER
# =========================================================

def optimize_memory():

    gc.collect()

    plt.close('all')

    print(
        "\nMemory Optimization Completed"
    )

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

            print(
                "\nDataset Selected Successfully"
            )

            return file_path

    except Exception as e:

        print(e)

    file_path = input(
        "\nEnter CSV Path: "
    ).strip()

    if os.path.exists(file_path):

        return file_path

    raise FileNotFoundError(
        f"\nFile Not Found:\n{file_path}"
    )

# =========================================================
# PRINT SECTION
# =========================================================

def print_section(title):

    print(
        f"\n{'=' * 20} "
        f"{title} "
        f"{'=' * 20}"
    )

# =========================================================
# DETECT TARGET COLUMN
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

# =========================================================
# DETECT CATEGORICAL COLUMNS
# =========================================================

def detect_categorical_columns(
    df,
    target_col
):

    categorical_cols = []

    for col in df.columns:

        if col == target_col:
            continue

        if df[col].dtype == "object":

            categorical_cols.append(col)

        elif df[col].nunique() <= 10:

            categorical_cols.append(col)

    return categorical_cols

# =========================================================
# GRAPH OUTPUT DIRECTORY
# =========================================================

def create_graph_output_folder(dataset_name):

    graph_dir = os.path.join(
        "outputs",
        "graphs",
        dataset_name.replace(".csv", "")
    )

    plt.close('all')

    optimize_memory()

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    for file in os.listdir(graph_dir):

        if file.endswith(".png"):

            file_path = os.path.join(
                graph_dir,
                file
            )

            try:

                os.remove(file_path)

            except Exception as e:

                print(
                    f"\nCould Not Delete: {file}"
                )

                print(e)

    return graph_dir

# =========================================================
# SAVE GRAPH FUNCTION
# =========================================================

def save_current_graph(
    graph_dir,
    graph_name
):

    try:

        os.makedirs(
            graph_dir,
            exist_ok=True
        )

        output_path = os.path.join(
            graph_dir,
            f"{graph_name}.png"
        )

        fig = plt.gcf()

        plt.draw()

        fig.tight_layout()

        dpi_value = 100 if FAST_MODE else 300

        fig.savefig(

            output_path,

            dpi=dpi_value,

            bbox_inches='tight',

            facecolor='white'
        )

        print(
            f"\nGraph Saved Successfully:"
            f"\n{output_path}"
        )

        plt.close(fig)

        optimize_memory()

    except Exception as e:

        print(
            "\nGraph Save Failed"
        )

        print(e)

# =========================================================
# PDF REPORT GENERATOR
# =========================================================

def generate_pdf_report(
    output_path,
    content
):

    doc = SimpleDocTemplate(
        output_path
    )

    styles = getSampleStyleSheet()

    story = []

    for line in content.split("\n"):

        safe_line = escape(line)

        story.append(

            Paragraph(
                safe_line,
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

    doc.build(story)

    print(
        "\nPDF Report Generated Successfully"
    )

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

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = load_dataset(file_path)

    optimize_memory()

    dataset_name = os.path.basename(
        file_path
    )

    # =====================================================
    # FAST MODE DATA REDUCTION
    # =====================================================

    if FAST_MODE and len(df) > 100000:

        print(
            "\nFAST MODE LARGE DATASET REDUCTION ENABLED"
        )

        df = df.sample(
            n=50000,
            random_state=42
        )

    # =====================================================
    # TARGET COLUMN DETECTION
    # =====================================================

    target_col = detect_target_column(df)

    print(
        f"\nTarget Column: {target_col}"
    )

    # =====================================================
    # GRAPH DIRECTORY
    # =====================================================

    graph_dir = create_graph_output_folder(
        dataset_name
    )

    # =====================================================
    # LOGGER
    # =====================================================

    set_log_file(dataset_name)

    write_log(
        f"Dataset Name: {dataset_name}"
    )

    write_log(
        f"Dataset Shape: {df.shape}"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    validate_dataset(df)

    # =====================================================
    # SUMMARY
    # =====================================================

    dataset_summary(df)

    # =====================================================
    # COPY BEFORE PROCESSING
    # =====================================================

    if len(df) > 50000:

        print(
            "\nUsing Sample Copy For Visualization"
        )

        df_before = df.sample(
            n=5000,
            random_state=42
        ).copy()

    else:

        df_before = df.copy()

    # =====================================================
    # PREPROCESSING
    # =====================================================

    df, encoders = preprocess_dataset(df)

    optimize_memory()

    print(
        "\nPreprocessing Completed"
    )

    # =====================================================
    # REMOVE EXTRA COLUMN
    # =====================================================

    if "data_type" in df.columns:

        df.drop(
            columns=["data_type"],
            inplace=True
        )

    # =====================================================
    # EDQS BEFORE
    # =====================================================

    edqs_before_metrics = calculate_edqs(
        df,
        target_col
    )

    edqs_before = (
        edqs_before_metrics["edqs"]
    )

    # =====================================================
    # IMBALANCE DETECTION
    # =====================================================

    imbalance_ratio, class_counts = (
        detect_imbalance(
            df,
            target_col
        )
    )

    imbalance_report(class_counts)

    # =====================================================
    # CATEGORICAL COLUMNS
    # =====================================================

    categorical_cols = (
        detect_categorical_columns(
            df,
            target_col
        )
    )

    categorical_cols = [
        col for col in categorical_cols
        if col in df.columns
    ]

    # =====================================================
    # SMOTE
    # =====================================================

    try:

        df = apply_smote(
            df,
            target_col,
            categorical_cols
        )

        optimize_memory()

    except Exception as e:

        print(e)

        write_log(str(e))

    # =====================================================
    # SYNTHETIC DATA
    # =====================================================

    try:

        if FAST_MODE:

            print(
                "\nFAST MODE ENABLED"
            )

            epochs = 5

        else:

            epochs = 50

        synthetic_df = generate_ctgan_data(
            df,
            target_col,
            epochs=epochs
        )

        max_rows = min(
            len(df),
            10000
        )

        if len(synthetic_df) > max_rows:

            synthetic_df = synthetic_df.sample(
                n=max_rows,
                random_state=42
            )

        df = pd.concat(

            [df, synthetic_df],

            ignore_index=True
        )

        optimize_memory()

    except Exception as e:

        print(e)

        write_log(str(e))

    # =====================================================
    # FAIRNESS
    # =====================================================

    fairness_results, fairness_score = (
        calculate_fairness(
            df,
            target_col
        )
    )

    # =====================================================
    # PRINT FAIRNESS RESULTS
    # =====================================================

    if isinstance(fairness_results, pd.DataFrame):

        if not fairness_results.empty:

            print(
                "\n========== FAIRNESS RESULTS ==========\n"
            )

            print(
                fairness_results.to_string(index=False)
            )

        else:

            print(
                "\nNo Bias Detected"
            )

    else:

        print(
            "\nNo Fairness Results Available"
        )

    # =====================================================
    # FAIRNESS FIX
    # =====================================================

    try:

        df = apply_fairness_fix(
            df,
            fairness_results,
            target_col
        )

        optimize_memory()

        print(
            "\nFairness Fix Applied"
        )

    except Exception as e:

        print(e)

        write_log(str(e))

    # =====================================================
    # SKEWNESS FIX
    # =====================================================

    try:

        df = fix_skewness(
            df,
            target_col
        )

        optimize_memory()

        print(
            "\nSkewness Fixed"
        )

    except Exception as e:

        print(e)

        write_log(str(e))

    # =====================================================
    # EDQS AFTER
    # =====================================================

    edqs_after_metrics = calculate_edqs(
        df,
        target_col
    )

    edqs_after = (
        edqs_after_metrics["edqs"]
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    print_section(
        "VISUALIZATION"
    )

    try:

        # BEFORE AFTER COUNTS

        plt.figure(figsize=(8, 5))

        plot_before_after_counts(
            df_before,
            df
        )

        save_current_graph(
            graph_dir,
            "before_after_counts"
        )

        # BOXPLOTS

        plt.figure(figsize=(10, 6))

        plot_boxplots(
            df_before,
            df
        )

        save_current_graph(
            graph_dir,
            "boxplots"
        )

        # HEATMAPS

        if not FAST_MODE:

            plt.figure(figsize=(10, 8))

            plot_heatmaps(
                df_before,
                df
            )

            save_current_graph(
                graph_dir,
                "heatmaps"
            )

            sns.reset_defaults()

        # PIECHARTS

        plt.figure(figsize=(10, 6))

        plot_piecharts(
            df_before,
            df,
            categorical_cols
        )

        save_current_graph(
            graph_dir,
            "piecharts"
        )

        # EDQS COMPARISON

        plt.figure(figsize=(8, 5))

        plot_edqs_comparison(
            edqs_before,
            edqs_after
        )

        save_current_graph(
            graph_dir,
            "edqs_comparison"
        )

        plt.close('all')

        optimize_memory()

        print(
            "\nAll Graphs Saved Successfully"
        )

    except Exception as e:

        print(
            "\nVisualization Error"
        )

        print(e)

        write_log(str(e))

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    print_section(
        "MODEL TRAINING"
    )

    try:

        if len(df) > 100000:

            print(
                "\nLarge Dataset Detected"
            )

            print(
                "\nApplying Training Sampling..."
            )

            df = df.sample(
                n=50000,
                random_state=42
            )

            print(
                f"\nReduced Training Shape: {df.shape}"
            )

        X = df.drop(
            columns=[target_col]
        )

        y = df[target_col]

        if y.nunique() > 1:

            stratify_option = y

        else:

            stratify_option = None

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,

                y,

                test_size=0.2,

                random_state=42,

                stratify=stratify_option
            )
        )

        model = train_model(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        metrics = evaluate_model(
            y_test,
            y_pred
        )

        print(
            "\n========== MODEL METRICS =========="
        )

        for key, value in metrics.items():

            print(
                f"{key} : {value:.4f}"
            )

        save_model(model)

        optimize_memory()

    except Exception as e:

        print(e)

        write_log(str(e))

    # =====================================================
    # SAVE FINAL DATASET
    # =====================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_path = os.path.join(
        "outputs",
        "final_responsible_ai_dataset.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFinal Dataset Saved:\n{output_path}"
    )

    # =====================================================
    # PDF REPORT
    # =====================================================

    report_content = f"""

RESPONSIBLE AI REPORT

====================================

Dataset:
{dataset_name}

EDQS BEFORE:
{edqs_before:.2f}

EDQS AFTER:
{edqs_after:.2f}

====================================

Graphs Folder:
{graph_dir}

"""

    generate_pdf_report(
        "outputs/responsible_ai_report.pdf",
        report_content
    )

    # =====================================================
    # FINAL CLEANUP
    # =====================================================

    plt.close('all')

    optimize_memory()

    # =====================================================
    # COMPLETED
    # =====================================================

    print_section(
        "RESPONSIBLE AI PIPELINE COMPLETED"
    )

# =========================================================
# RUN PIPELINE
# =========================================================

if __name__ == "__main__":

    main()
