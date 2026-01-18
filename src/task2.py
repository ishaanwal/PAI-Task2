from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Supermarket_dataset_PAI.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
