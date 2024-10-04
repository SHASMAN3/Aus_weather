import pandas as pd
import numpy as np
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import svm


from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# from keras.models import Sequential

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

# function for preprocessing data
def preprocess_dataframe(df):
    df_filtered=df[df['RainTomorrow'].notna()]
    df_inference=df[df['RainTomorrow'].isna()]
    return df_filtered, df_inference

def process_date_column(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', dayfirst=True)
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df.drop(date_column, axis=1, inplace=True)
    return df

def impute_missing_values(df):
    """Imputes missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to impute missing values in.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    columns_with_null = df.isnull().sum()
    columns_with_null = columns_with_null[columns_with_null > 0]

    for column in columns_with_null.index:
        if df[column].dtypes == 'object':
            # Check if there are any duplicate values in the column
            if df[column].nunique() > 1:
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
            else:
                # If there are no duplicate values, fill with the only unique value
                df[column].fillna(df[column].unique()[0], inplace=True)
        elif df[column].dtypes == 'float64':
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

    return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

"""def fit_label_encoders(df: pd.DataFrame, save_path: str) -> (pd.DataFrame, dict):
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'RainTomorrow':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    # Save label_encoders to disk
    with open(save_path, 'wb') as f:
        pickle.dump(label_encoders, f)

    return df, label_encoders

def load_label_encoders(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def transform_label_data(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col].astype(str))
    return df

def load_and_transform_new_data(df: pd.DataFrame, encoder_path: str) -> pd.DataFrame:
    label_encoders = load_label_encoders(encoder_path)
    df_transformed = transform_label_data(df, label_encoders)
    return df_transformed

# Example usage
df_encoded, LE = fit_label_encoders(df2, 'label_encoder.pkl')
df_inference_encoded = load_and_transform_new_data(df_inference, 'label_encoder.pkl')"""

def transform_data(df:pd.DataFrame, label_encoders:dict) -> pd.DataFrame:
    for column, le in label_encoders.items():
        df[column] = df[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return df

def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != 'RainTomorrow':
            df[col] = df[col].astype(float)
    return df


def drop_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate 0.5th and 99.5th percentiles for each numeric column
    Q01 = numeric_df.quantile(0.005)
    Q99 = numeric_df.quantile(0.995)

    # Define lower and upper bounds for numeric columns
    lower_bound = Q01 - 1.5 * (Q99 - Q01)
    upper_bound = Q99 + 1.5 * (Q99 - Q01)

    # Identify outliers for each numeric column
    outliers = pd.DataFrame(False, index=df.index, columns=numeric_df.columns)
    for col in numeric_df.columns:
        outliers[col] = (numeric_df[col] < lower_bound[col]) | (numeric_df[col] > upper_bound[col])

    # Remove rows where any numeric column has outliers
    filtered_df = df[~outliers.any(axis=1)]
    return filtered_df

def min_max_scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()

    # separate the feature from the target column
    features = df.drop('RainTomorrow', axis=1)
    target = df['RainTomorrow']

    # scale the feature
    scaled_features = scaler.fit_transform(features)

    # reassemble the dataframe
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    scaled_df['RainTomorrow'] = target

    return scaled_df

    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import sys
import warnings
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def split_feature_target(df,target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def load_model(logreg_model):
    # logreg_model= catalog.load("logreg,pkl")
    return logreg_model

def predict_and_update(X_infer,df_inference,model,target_column):
    y_pred_infer=model.predict(X_infer)
    df_inference["RainTomorrow_predicted"]=y_pred_infer
    df_inference.drop(target_column,axis=1,inplace=True)
    return df_inference