import os,gc,warnings,tkinter as tk
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape
from modules.dataset.loader import load_dataset
from modules.dataset.validator import validate_dataset
from modules.dataset.summary import dataset_summary
from modules.preprocessing.preprocessing_pipeline import preprocess_dataset
from modules.preprocessing.skewness_fixer import fix_skewness
from modules.balancing.imbalance_detector import detect_imbalance
from modules.balancing.imbalance_report import imbalance_report
from modules.balancing.smote_module import apply_smote
from modules.synthetic.ctgan_generator import generate_ctgan_data
from modules.bias.fairness_metrics import calculate_fairness
from modules.bias.fairness_fix import apply_fairness_fix
from modules.metrics.edqs import calculate_edqs
from modules.metrics.eri import calculate_eri
from modules.metrics.rai import calculate_rai
from modules.model.train_model import train_model
from modules.model.evaluate_model import evaluate_model
from modules.model.save_model import save_model
from modules.visualization.before_after_graphs import plot_before_after_counts
from modules.visualization.boxplots import plot_boxplots
from modules.visualization.piecharts import plot_piecharts
from modules.visualization.edqs_graph import plot_edqs_comparison
from modules.visualization.bias_visualizations import (bias_bar_graph, bias_heatmap, fairness_comparison_chart)
from modules.utils.save_metrics import save_metrics
from modules.utils.logger import set_log_file,write_log
from modules.llm.llm_decision_engine import (generate_llm_analysis, generate_llm_recommendations)
FAST_MODE=True
TEST_SIZE=0.20
RANDOM_STATE=42
os.makedirs("outputs",exist_ok=True)
METRICS_TEXT_PATH=os.path.join("outputs", "all_metrics_log.txt")
with open(METRICS_TEXT_PATH,"w",encoding="utf-8") as f: f.write("RESPONSIBLE AI METRICS LOG\n\n")
def optimize_memory():
    gc.collect()
    plt.close("all")
def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")
def append_metrics_to_txt(title,data):
    with open(METRICS_TEXT_PATH,"a",encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{title}\n{'='*60}\n\n")
        if isinstance(data,dict):
            for k,v in data.items():
                f.write(f"{k} : {v}\n")
        else:
            f.write(str(data))
        f.write("\n")
def choose_file():
    try:
        root=tk.Tk()
        root.withdraw()
        root.attributes('-topmost',True)
        file_path=filedialog.askopenfilename(title="Select CSV Dataset", filetypes=[("CSV Files","*.csv")])
        root.destroy()
        if file_path:
            print("\nDataset Selected Successfully")
            return file_path
    except Exception as e:
        print(e)
    file_path=input("\nEnter CSV Path: ").strip()
    if os.path.exists(file_path):
        return file_path
    raise FileNotFoundError(file_path)
def detect_target_column(df):
    print("\nDetecting Target Column...")
    target_keywords=["target","label","class","output","result","response","status","prediction","survived","default","churn","risk","score","rating"]
    excluded_keywords=["id","serial","phone","email","timestamp","date","time"]
    scores={}
    total_rows=len(df)
    for col in df.columns:
        score=0
        lower_col=col.lower()
        unique_values=df[col].nunique()
        unique_ratio=unique_values/max(total_rows,1)
        dtype=str(df[col].dtype)
        if unique_values<=1:
            continue
        if any(x in lower_col for x in excluded_keywords):
            score-=100
        if any(x in lower_col for x in target_keywords):
            score+=60
        if col==df.columns[-1]:
            score+=30
        if dtype=="object":
            score+=20
        if unique_values<=10:
            score+=25
        if unique_ratio>0.95:
            score-=40
        scores[col]=score
    sorted_scores=sorted(scores.items(),key=lambda x:x[1],reverse=True)
    print("\n========== TARGET COLUMN SCORES ==========")
    for col,score in sorted_scores:
        print(f"{col} --> {score}")
    target_col=sorted_scores[0][0]
    print(f"\nDetected Target Column : {target_col}")
    return target_col
def detect_categorical_columns(df,target_col):
    return [col for col in df.columns if col!=target_col and (str(df[col].dtype)=="object" or "category" in str(df[col].dtype))]
def create_graph_output_folder(dataset_name):
    graph_dir=os.path.join("outputs","graphs", dataset_name.replace(".csv",""))
    os.makedirs(graph_dir,exist_ok=True)
    return graph_dir
def save_current_graph(graph_dir,graph_name):

    try:

        output_path=os.path.join(
            graph_dir,
            f"{graph_name}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight"
        )

        print(f"\nGraph Saved Successfully:\n{output_path}")

        plt.close()

    except Exception as e:
        print(e)


def generate_pdf_report(output_path,content):

    doc=SimpleDocTemplate(output_path)
    styles=getSampleStyleSheet()
    story=[]

    for line in content.split("\n"):

        story.append(
            Paragraph(
                escape(line),
                styles["BodyText"]
            )
        )

        story.append(Spacer(1,8))

    doc.build(story)

    print("\nPDF Report Generated Successfully")


def safe_remove_duplicates(df):

    before=len(df)

    duplicate_percent=(
        df.duplicated()
        .mean()
    )*100

    if duplicate_percent>15:

        df=df.drop_duplicates()

        print(
            f"\nDuplicate Rows Removed : "
            f"{before-len(df)}"
        )

    else:

        print(
            "\nDuplicate Removal Skipped "
            "(safe duplicate level)"
        )

    return df

def validate_final_dataset(
    df,
    target_col,
    task_type
):

    if len(df)==0:
        raise ValueError(
            "Dataset became empty"
        )

    if target_col not in df.columns:
        raise ValueError(
            "Target column missing"
        )
    print("\n========== TARGET CHECK ==========")
    print(
        df[target_col]
        .value_counts(dropna=False)
    )

    unique_classes=(
        df[target_col]
        .nunique()
    )

    print(
        f"\nUnique Classes : "
        f"{unique_classes}"
    )

    if task_type=="classification":

        if unique_classes<2:

            raise ValueError(
                "Classification requires "
                "minimum 2 classes"
            )


def main():

    print_section("RESPONSIBLE AI PIPELINE STARTED")

    file_path=choose_file()

    df=load_dataset(file_path)

    dataset_name=os.path.basename(file_path)

    graph_dir=create_graph_output_folder(dataset_name)

    set_log_file(dataset_name)

    optimize_memory()

    validate_dataset(df)

    dataset_summary(df)

    target_col=detect_target_column(df)

    print(f"\nTarget Column : {target_col}")

    # =====================================
    # TASK TYPE DETECTION
    # =====================================

    unique_count=(
        df[target_col]
        .nunique()
    )

    total_rows=len(df)

    if (
        pd.api.types.is_numeric_dtype(
            df[target_col]
        )
        and (
            unique_count>15
            or unique_count/total_rows>0.05
        )
    ):

        task_type="regression"

    else:

        task_type="classification"

    print(
        f"\nTask Type : "
        f"{task_type.upper()}"
    )

    df_before=df.copy()

    target_series=(
    df[target_col]
    .copy()
    .reset_index(drop=True)
)

    features_df=df.drop(
    columns=[target_col]
)

    features_df,encoders=preprocess_dataset(
    features_df
)

    features_df=features_df.reset_index(
    drop=True
)

    min_len=min(
    len(features_df),
    len(target_series)
)

    features_df=features_df.iloc[:min_len]

    target_series=target_series.iloc[:min_len]

    df=pd.concat(
    [features_df,target_series],
    axis=1
)

    optimize_memory()

    validate_final_dataset(
    df,
    target_col,
    task_type
)

    print_section("INITIAL METRICS")

    edqs_before_metrics=calculate_edqs(df,target_col, task_type)

    edqs_before=edqs_before_metrics["edqs"]

    if task_type=="classification":

        imbalance_ratio_before,class_counts_before=(
            detect_imbalance(
                df,
                target_col
            )
        )

        imbalance_report(
            class_counts_before
        )

    else:

        imbalance_ratio_before=0

        class_counts_before={}

    if task_type=="classification":

        bias_results_before,fairness_before=(
            calculate_fairness(df,target_col)
        )

    else:

        bias_results_before=[]
        fairness_before=1.0

    print(f"\nFairness Before : {fairness_before:.2f}")

    X=df.drop(columns=[target_col])
    y=df[target_col]

    stratify_option=None

    if task_type=="classification":

        class_counts=(
            y.value_counts()
        )

        min_class_count=(
            class_counts.min()
        )

        print(
            "\n========== STRATIFY CHECK =========="
        )

        print(class_counts)

        if min_class_count>=2:

            stratify_option=y

            print(
                "\nStratify Enabled"
            )

        else:

            print(
                "\nStratify Disabled "
                "(very small class detected)"
            )

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_option
    )

    train_df=pd.concat([X_train,y_train],axis=1)

    categorical_cols=[]

    for col in train_df.columns:

        if col!=target_col:

            unique_count=(
                train_df[col]
                .nunique()
            )

            if unique_count<=10:

                categorical_cols.append(
                    col
                )

    if task_type=="classification":

        try:

            imbalance_ratio_train,_=detect_imbalance(
                train_df,
                target_col
            )

            class_counts=(
                train_df[target_col]
                .value_counts()
            )

            min_class_count=(
                class_counts.min()
            )

            if (
                imbalance_ratio_train<0.80
                and min_class_count>=2
            ):

                smote_output=apply_smote(
    train_df,
    target_col,
    categorical_cols
)

                if smote_output is not None:

                    train_df=smote_output

                    print(
        "\nSMOTE Applied "
        "On Training Data"
    )

                else:

                    print(
        "\nSMOTE Failed"
    )

            else:

                print(
                    "\nSMOTE Skipped "
                    "(very small class detected)"
                )

        except Exception as e:
            print(e)
            write_log(str(e))

    try:

        if (

            task_type=="classification"

            and

            len(train_df)<5000

            and

            imbalance_ratio_before<0.50
        ):

            ctgan_df=train_df.copy()

            for col in ctgan_df.select_dtypes(
                include=["object"]
            ).columns:

                ctgan_df[col]=(
                    ctgan_df[col]
                    .astype("category")
                    .cat.codes
                )

            synthetic_df=generate_ctgan_data(
                ctgan_df,
                target_col,
                epochs=5
            )

            synthetic_df=(
                synthetic_df
                .dropna()
                .drop_duplicates()
            )

            if task_type=="regression":

                original_min=(
                    df[target_col]
                    .min()
                )

                original_max=(
                    df[target_col]
                    .max()
                )

                synthetic_df[target_col]=(

                    synthetic_df[target_col]

                    .clip(
                        original_min,
                        original_max
                    )
                )

                synthetic_df[target_col]=(

                    synthetic_df[target_col]

                    .round()

                    .astype(int)
                )

            if len(synthetic_df)>0:

                train_df=pd.concat(
                    [
                        train_df,
                        synthetic_df
                    ],
                    ignore_index=True
                )

                print(
                    f"\nSynthetic Rows Added : "
                    f"{len(synthetic_df)}"
                )

        else:

            print(
                "\nCTGAN Skipped "
                "(dataset too large or imbalance low)"
            )

    except Exception as e:

        print(e)

        write_log(str(e))
        
        
        
    train_df=safe_remove_duplicates(train_df)
    
    if task_type=="classification":

        try:

            train_df=apply_fairness_fix(
                train_df,
                bias_results_before,
                target_col
            )

            print(
                "\nFairness Fix Applied Successfully"
            )

        except Exception as e:

            print(e)

            write_log(str(e))

    try:

        train_df=fix_skewness(
            train_df,
            target_col
        )

    except Exception as e:
        print(e)

    validate_final_dataset(train_df,target_col, task_type)

    print_section("FINAL FAIRNESS METRICS")

    if task_type=="classification":

        bias_results_after,fairness_after=(
            calculate_fairness(
                train_df,
                target_col
            )
        )
    else:
        bias_results_after=[]
        fairness_after=1.0
    print(
        f"\nFairness After : "
        f"{fairness_after:.2f}"
    )

    edqs_after_metrics=calculate_edqs(train_df,target_col, task_type)

    edqs_after=edqs_after_metrics["edqs"]

    if task_type=="classification":

        imbalance_ratio_after,class_counts_after=(
            detect_imbalance(
                train_df,
                target_col
            )
        )

    else:

        imbalance_ratio_after=0

        class_counts_after={}
        
    if task_type=="classification":

        print(
            "\n========== FINAL CLASS CHECK =========="
        )

        print(
            train_df[target_col]
            .value_counts()
        )

    if task_type=="classification":

        if (
            train_df[target_col]
            .nunique()<2
        ):

            raise ValueError(
                "\nModel training stopped "
                "because only one class "
                "is available"
            )
            
    X_train_final=train_df.drop(columns=[target_col])
    y_train_final=train_df[target_col]

    model=train_model(
        X_train_final,
        y_train_final,
        task_type
    )

    y_pred=model.predict(X_test)

    metrics=evaluate_model(
        y_test,
        y_pred,
        task_type
    )

    eri_before=calculate_eri(
        fairness_score=fairness_before,
        accuracy=0.50,
        imbalance_ratio=imbalance_ratio_before,
        explainability_score=0.50
    )

    eri_after=calculate_eri(

        fairness_score=
        fairness_after,

        accuracy=(

            metrics.get(
                "accuracy",
                0
            )

            if task_type=="classification"

            else

            metrics.get(
                "r2",
                0
            )
        ),

        imbalance_ratio=
        imbalance_ratio_after,

        explainability_score=0.85
    )

    rai_before=calculate_rai(
        edqs_before,
        fairness_before,
        0.50,
        eri_before
    )

    rai_after=calculate_rai(
        edqs_after,
        fairness_after,
        metrics.get("accuracy",0),
        eri_after
    )

    print_section("FINAL METRICS")

    print(f"\nEDQS Before : {edqs_before:.2f}")
    print(f"EDQS After : {edqs_after:.2f}")

    print(f"\nFairness Before : {fairness_before:.2f}")
    print(f"Fairness After : {fairness_after:.2f}")

    print(f"\nERI Before : {eri_before:.4f}")
    print(f"ERI After : {eri_after:.4f}")

    print(f"\nRAI Before : {rai_before:.2f}")
    print(f"RAI After : {rai_after:.2f}")

    try:

        plt.figure(figsize=(8,5))
        plot_before_after_counts(df_before,train_df)
        save_current_graph(graph_dir,"before_after_counts")

        plt.figure(figsize=(8,5))
        plot_edqs_comparison(edqs_before,edqs_after, task_type)
        save_current_graph(graph_dir,"edqs_comparison")

        fairness_comparison_chart(
            fairness_before,
            fairness_after,
            graph_dir
        )

        if (
            task_type=="classification"
            and len(bias_results_before)>0
        ):

            bias_bar_graph(
                bias_results_before,
                graph_dir
            )

        bias_heatmap(train_df,graph_dir)

    except Exception as e:
        print(e)

    metrics_data={
        "EDQS Before":edqs_before,
        "EDQS After":edqs_after,
        "Fairness Before":fairness_before,
        "Fairness After":fairness_after,
        "ERI Before":eri_before,
        "ERI After":eri_after,
        "RAI Before":rai_before,
        "RAI After":rai_after
    }

    if task_type=="classification":

        metrics_data.update({

            "Accuracy":metrics.get(
                "accuracy",
                0
            ),

            "Precision":metrics.get(
                "precision",
                0
            ),

            "Recall":metrics.get(
                "recall",
                0
            ),

            "F1 Score":metrics.get(
                "f1_score",
                0
            )
        })

    else:

        metrics_data.update({

            "MAE":metrics.get(
                "mae",
                0
            ),

            "MSE":metrics.get(
                "mse",
                0
            ),

            "RMSE":metrics.get(
                "rmse",
                0
            ),

            "R2 Score":metrics.get(
                "r2",
                0
            )
        })

    append_metrics_to_txt("FINAL METRICS",metrics_data)

    metrics_path=save_metrics(metrics_data)

    with open(metrics_path,"r",encoding="utf-8") as f:
        metrics_text=f.read()

    llm_analysis=generate_llm_analysis(metrics_text)
    print("\n========== LLM ANALYSIS ==========")
    print(llm_analysis)
    recommendations=generate_llm_recommendations(metrics_text)
    print("\n========== LLM RECOMMENDATIONS ==========")
    print(recommendations)
    save_model(model)
    output_path=os.path.join(
        "outputs",
        "final_responsible_ai_dataset.csv"
    )
    train_df.to_csv(output_path,index=False)
    print(f"\nFinal Dataset Saved:\n{output_path}")
    report_content=f"""
RESPONSIBLE AI REPORT
Dataset : {dataset_name}
EDQS Before : {edqs_before:.2f}
EDQS After : {edqs_after:.2f}
Fairness Before : {fairness_before:.2f}
Fairness After : {fairness_after:.2f}
ERI Before : {eri_before:.4f}
ERI After : {eri_after:.4f}
RAI Before : {rai_before:.2f}
RAI After : {rai_after:.2f}
{
f'''
Accuracy : {metrics.get("accuracy",0):.4f}
Precision : {metrics.get("precision",0):.4f}
Recall : {metrics.get("recall",0):.4f}
F1 Score : {metrics.get("f1_score",0):.4f}
'''
if task_type=="classification"
else
f'''
MAE : {metrics.get("mae",0):.4f}
MSE : {metrics.get("mse",0):.4f}
RMSE : {metrics.get("rmse",0):.4f}
R2 Score : {metrics.get("r2",0):.4f}
'''
}
LLM ANALYSIS:
{llm_analysis}
"""
    generate_pdf_report(
        "outputs/responsible_ai_report.pdf",
        report_content
    )
    optimize_memory()
    print_section("RESPONSIBLE AI PIPELINE COMPLETED")
if __name__=="__main__":
    main()

