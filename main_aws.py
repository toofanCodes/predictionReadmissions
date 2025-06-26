from Extraction_aws import extraction
from Transformation_aws import transforming
from Training_aws import run_training  # If you have S3 support in Training, else use Training

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')

def main():
    # Ask user for the S3 or local data file path
    data_file = input("Enter the S3 path (s3://bucket/key) or local path to the data CSV file: ").strip()
    if not data_file:
        print("No file provided. Exiting.")
        return

    # Step 1: Extraction (now supports S3)
    staging_table = extraction(data_file)
    if not staging_table:
        print("Extraction failed or no new data to process.")
        return

    # Step 2: Transformation (now supports S3)
    feature_table = transforming(staging_table)
    if not feature_table:
        print("Transformation failed.")
        return

    # Step 3: Training (prediction by default, retrain with update=1)
    run_training(feature_table)  # prediction only
    # run_training(feature_table, update=1)  # retrain and predict

if __name__ == "__main__":
    main()
