from Extraction import extraction
from Transformation import transforming
from Training import run_training
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')

def main():
    # Ask user for the data file path
    data_file = input("Enter the path to the data CSV file: ").strip()
    if not data_file:
        print("No file provided. Exiting.")
        return

    # Step 1: Extraction
    staging_table = extraction(data_file)
    if not staging_table:
        print("Extraction failed or no new data to process.")
        return

    # Step 2: Transformation
    feature_table = transforming(staging_table)
    if not feature_table:
        print("Transformation failed.")
        return

    # Step 3: Training (prediction by default, retrain with update=1)
    run_training(feature_table)  # prediction only
    # run_training(feature_table, update=1)  # retrain and predict

if __name__ == "__main__":
    main()
