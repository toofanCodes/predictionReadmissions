from dotenv import load_dotenv
from pathlib import Path
from s3_utils import write_df_to_s3

load_dotenv(dotenv_path=Path('.') / '.env')

import pandas as pd
from SQLEngineConnector import connectToDB

def transforming(table_name):
    engine = connectToDB()
    if engine is None:
        print("Database connection failed. Check your .env and DB settings.")
        return
    print(f"\n \n «--- Data Transformation started ---»")
    
    def extractData(tabName):
        query = f"SELECT * FROM {tabName}"
        try:
            df = pd.read_sql(query, con=engine)
            print(f"» {tabName} dataset loaded - {df.shape[0]} rows")
            df.replace('?', pd.NA, inplace=True)
            nullDf = round((df.isna().sum()*100)/len(df), 2)
            print(f"» {tabName}:- null values in %:\n", nullDf[nullDf > 0])
            print(f"\n")
            return df
        except Exception as e:
            print(f"Failed to load {tabName}: {e}")
            return None

    def checkNulls(df):
        nullDf = round((df.isna().sum()*100)/len(df), 2)
        nullDf = nullDf[nullDf > 0].sort_values()
        if nullDf.empty:
            print("No null values found.")
        else:
            print(f"» null values in %:\n",)
            print(nullDf, "\n")

    df_main = extractData(table_name)
    if df_main is None:
        print("Aborting transformation due to data load failure.")
        return

    # Nulls in max_glu_serum and A1Cresult were replaced with -1 for ingestion, lets remove those now
    df_main['max_glu_serum'] = df_main['max_glu_serum'].replace('-1', pd.NA)
    df_main['A1Cresult'] = df_main['A1Cresult'].replace('-1', pd.NA)

    print("checking nulls before cleaning data")
    checkNulls(df_main)

    def declutter(df_main):
        df_main = (df_main.assign(
                        readmittedStatus=df_main['readmitted'].apply(lambda x: 1 if x == '<30' else 0),
                        age_bucket=df_main['age'].map({'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                                                       '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                                                       '[80-90)': 85, '[90-100)': 95})
                    )
                    .drop(['readmitted', 'weight', 'max_glu_serum', 'A1Cresult'], axis=1)
                    .fillna('Unknown')
                )
        df_main[['diag_1', 'diag_2', 'diag_3']] = df_main[['diag_1', 'diag_2', 'diag_3']].replace(pd.NA, 'Unknown')
        print("data cleaning complete")
        return df_main

    df_main = declutter(df_main)
    checkNulls(df_main)

    try:
        df_main.to_sql('feature_admissions', 
                       con=engine, 
                       if_exists='replace', 
                       index=False,
                       method='multi')
        print("Saved to feature_admissions")
        
        # Save to S3 as well
        bucket = 'readmission-risk-data'
        key = 'processed/feature_admissions.csv'
        write_df_to_s3(df_main, bucket, key)
        print(f"Saved feature_admissions to s3://{bucket}/{key}")

    except Exception as e:
        print(f"Failed to write to feature_admissions: {e}")
        return None

    return 'feature_admissions'
