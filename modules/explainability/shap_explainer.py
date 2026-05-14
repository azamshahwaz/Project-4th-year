import os

import shap

import matplotlib.pyplot as plt

def generate_shap(

    model,

    X
):

    os.makedirs(
        "outputs/graphs",
        exist_ok=True
    )

    explainer = (
        shap.TreeExplainer(model)
    )

    shap_values = (
        explainer.shap_values(X)
    )

    shap.summary_plot(

        shap_values,

        X,

        show=False
    )

    plt.savefig(

        "outputs/graphs/shap_summary.png",

        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSHAP Summary Plot Saved"
    )