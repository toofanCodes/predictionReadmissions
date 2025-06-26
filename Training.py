import os
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
import numpy as np
from SQLEngineConnector import connectToDB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import joblib
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
from sqlalchemy import text


engine = connectToDB()

load_dotenv(dotenv_path=Path('.') / '.env')

warnings.filterwarnings('ignore', category=RuntimeWarning)

MODEL_FILE = "xgboost_model.pkl"

def create_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        [
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ],
        remainder='drop'
    )

def create_xgboost_pipeline(numeric_cols, categorical_cols):
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.005,
            n_estimators=500,
            min_child_weight=2,
            gamma=1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=2025,
            n_jobs=-1,
            eval_metric='auc',
            enable_categorical=False
        ))
    ])
    return pipeline

def plot_curves(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.subplot(1,2,2)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def print_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps['model']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

def train_model(table_name):
    engine = connectToDB()
    if engine is None:
        print("Database connection failed.")
        return
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
    X = df.drop(columns=['readmittedStatus', 'encounter_id', 'patient_nbr'])
    y = df['readmittedStatus']

    # Handle high/low cardinality
    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() > 10:
            top_codes = X[col].value_counts().nlargest(6).index
            X[col] = X[col].where(X[col].isin(top_codes), 'Other')

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2025
    )

    pipeline = create_xgboost_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    plot_curves(y_test, y_proba)
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    print_feature_importance(pipeline, feature_names)

    # Save model
    model_info = {
        'pipeline': pipeline,
        'feature_names': list(X_train.columns),
        'training_date': pd.Timestamp.now()
    }
    joblib.dump(model_info, MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")

def predict_with_model(table_name, save_to_db=True):
    engine = connectToDB()
    if engine is None:
        print("Database connection failed.")
        return
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
    X = df.drop(columns=['readmittedStatus', 'encounter_id', 'patient_nbr'])
    model_info = joblib.load(MODEL_FILE)
    pipeline = model_info['pipeline']
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    df['prediction'] = preds
    df['probability'] = proba
    print(df[['prediction', 'probability']].head())

    # Save predictions to a new table
    if save_to_db:
        pred_table = f"predictions_{table_name}"
        # Optionally, drop if exists to avoid errors
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{pred_table}`;"))
        df[['encounter_id', 'patient_nbr', 'prediction', 'probability']].to_sql(
            pred_table,
            con=engine,
            if_exists='replace',
            index=False,
            method='multi'
        )
        print(f"Predictions saved to table: {pred_table}")

    return df[['prediction', 'probability']]

def run_training(table_name, update=0, drop_temp=True):
    if update == 1:
        print("Retraining model...")
        train_model(table_name)
    else:
        print(f"\n\n «--- Running prediction using existing model---»")
        predict_with_model(table_name)
        if drop_temp:
            drop_table(table_name)

def drop_table(table_name):
    engine = connectToDB()
    if engine is None:
        print("Database connection failed. Cannot drop table.")
        return
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`;"))
            # print(f"Table {table_name} dropped.")
        except Exception as e:
            print(f"Failed to drop table {table_name}: {e}")
