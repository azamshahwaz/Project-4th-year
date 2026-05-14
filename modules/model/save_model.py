import os
import joblib

def save_model(model):

    os.makedirs(
        "outputs/models",
        exist_ok=True
    )

    joblib.dump(

        model,

        "outputs/models/trained_model.pkl"
    )

    print(
        "\nModel Saved Successfully"
    )