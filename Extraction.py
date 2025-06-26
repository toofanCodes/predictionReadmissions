from dotenv import load_dotenv
from pathlib import Path
import os
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
import json
import hashlib

def file_hash(path):
    df = pd.read_csv(path)
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def is_new_data(path, reference_path="data/diabetic_data.csv"):
    try:
        return file_hash(path) != file_hash(reference_path)
    except Exception as e:
        print(f"Comparison failed: {e}")
        return True

def extraction(path, typeOfOp=0):
    
    def createSchema(path, typeOfOp=0):
        df = pd.read_csv(path)
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            nullable = df[col].isnull().any()
            schema[col] = {
                "dtype": dtype,
                "nullable": bool(nullable)
            }
        metadata = {
            "fileName": path,
            "createdAt": datetime.now().isoformat(),
            "numColumns": len(df.columns),
            "numRows": len(df)
        }
        schema_cleaned = {
            "metadata": metadata,
            "schema": schema
        }

        def compareSchema(filename):
            try:
                with open(filename) as newFile, open("schemaInput/schema.json") as baseFile:
                    jNew = json.load(newFile)
                    jBase = json.load(baseFile)
            except Exception as e:
                print(f"Error reading schema files: {e}")
                return False

            num_colsNew = jNew["metadata"].get("numColumns")
            num_colsBase = jBase["metadata"].get("numColumns")
            colsNew = set(jNew["schema"].keys())
            colsBase = set(jBase["schema"].keys())

            if num_colsNew != num_colsBase:
                direction = "additional columns found" if num_colsNew > num_colsBase else "columns are missing"
                mismatch = abs(num_colsNew - num_colsBase)
                print(f"Mismatch in number of columns: {mismatch} {direction}")
                print(f"Columns in new file not in base: {colsNew - colsBase}")
                print(f"Columns in base not in new file: {colsBase - colsNew}")
                return False

            if colsNew != colsBase:
                print(f"Column name mismatch:")
                print(f"Columns in new file not in base: {colsNew - colsBase}")
                print(f"Columns in base not in new file: {colsBase - colsNew}")
                return False
            else:
                print("Column names match.")

            mismatches = []
            for col in colsNew:
                dt1 = jNew["schema"][col]["dtype"]
                dt2 = jBase["schema"][col]["dtype"]
                if dt1 != dt2:
                    mismatches.append((col, dt1, dt2))
            if mismatches:
                print("Dtype mismatches exist:")
                for col, d1, d2 in mismatches:
                    print(f"- {col}: {d1} vs {d2}")
                return False
            else:
                print("Dtypes match.")
                return True

        if typeOfOp == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"schemaInput/schema_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(schema_cleaned, f, indent=4)
            print("Schema ready to be compared")
            if compareSchema(filename):
                print("»»»»» continue with extraction..")
                return True
            else:
                print("[STOP] address issues and retry...!")
                return False
        elif typeOfOp == 1:
            with open('schemaInput/schema.json', "w") as f:
                json.dump(schema_cleaned, f, indent=4)
            print("Schema structure updated")

    if not createSchema(path, typeOfOp):
        exit(1) # some error occured in handling new data file
    else: # new data file is inline with req
        print(f"\n\n «--- Data Extraction started ---»")
        
        # Compare with reference data
        if not is_new_data(path, "data/diabetic_data.csv"):
            print("No changes detected. Skipping table creation and ingestion.")
            return
        
        # Connecting to local database, later on to be repurposed to connect to AWS
        def get_connection():
            env_path = Path(".") / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                print(".env loaded")
            else:
                print(f" .env file NOT found at {env_path}")

            try:
                conn = pymysql.connect(
                    host=os.getenv("DB_HOST"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS"),
                    database=os.getenv("DB_NAME")
                )
                print("Database connection established.")
                return conn
            except pymysql.MySQLError as e:
                print(f"Database connection error: {e}")
                return None
            
        # Add this helper function
        def get_unique_table_name(base_name="staging_admissions"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}"
            
        # Add this helper function
        def clone_table_structure(conn, source_table, new_table):
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE TABLE IF NOT EXISTS `{new_table}` LIKE `{source_table}`;")
            conn.commit()
            
        # Setup and test connection
        conn= get_connection()
        cursor = conn.cursor()
        if conn:
            print("Setup success - connection established")
            if cursor:
                print("cursor link success")
            else:
                print("cursor link failed")
        else:
            print("Setup failed")
            
        # Above process has to be implented for each table, and we have total 4
        # so lets make a funciton out of it

        def extractAndIngest(df, dimName, conn):
            
            print(f"loaded {dimName} with a length of {len(df)}")
            
            columnNames = df.columns.tolist()
            columnNames_fmt = ", ".join([f"`{col}`" for col in columnNames])
            placeholders = ", ".join(['%s'] * len(columnNames))
            
            queryStatement = f"""REPLACE INTO {dimName} ({columnNames_fmt})
                            VALUES ({placeholders})"""
            # query statement is ready
            
            values = list(df.itertuples(index = False, name = None))
            # values loaded for enabling .executemany()
            
            try:
                with conn.cursor() as cursor:
                    for start in range(0, len(values), 10000):
                        batch = values[start: start + 10000]
                        cursor.executemany(queryStatement, batch)
                        # ingest in batches of 1000
                        
                        conn.commit()
                        # commit the changes
                        
                        # print(f"Inserted rows into {dimName} from {start} to {start+1000}")
                    cursor.execute(f"SELECT COUNT(*) FROM {dimName}")
                    # result of query is stores in cursor's internal state
                    
                    print(f"{datetime.now()} Total rows in {dimName}:", cursor.fetchone()[0])
                    # we use .fetchone() to get the next row, in our case only row as a tuple
                    # [0] extracts the last value in the tuple - total rows field
                    # similarly fetchall() gets all rows, fetchmany(n) gets n rows

            except Exception as e:
                print(f"{datetime.now()} Failed loading {dimName}: {e}")
                conn.rollback()
                # rollback all changes, if there's an error midway
            
        # Generate unique table name and clone structure
        new_table_name = get_unique_table_name("staging_admissions")
        clone_table_structure(conn, "staging_admissions", new_table_name)

        rawDf = pd.read_csv(path)
        rawDf = rawDf.where(pd.notnull(rawDf), None)  # Convert NaN to None
        print("ETL process started")

        # print(rawDf['max_glu_serum'].value_counts())
        # print(rawDf['A1Cresult'].value_counts())

        rawDf = rawDf.fillna({
            'max_glu_serum': -1,
            'A1Cresult': -1
        })

        rawDf.replace('?', np.nan, inplace=True)
        rawDf = rawDf.where(pd.notnull(rawDf), None)  # Convert NaN to None again after replacements

        extractAndIngest(rawDf, new_table_name, conn)
        
        cursor.close()
        conn.close()
        print("connection closed")
        return new_table_name  # <-- Add this line

