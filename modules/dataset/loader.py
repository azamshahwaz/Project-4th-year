import pandas as pd


def load_dataset(file_path):

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df