from typing import Dict, Tuple
import pandas as pd
import os 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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


# Function to drop missing values in a DataFrame
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

# LABEL ENCODER
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

from sklearn.preprocessing import LabelEncoder

def label_encode_object_column(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = df[column].astype(str)  # Ensure all data is string type
        le.fit(df[column])
        label_encoders[column] = le
        df[column] = le.transform(df[column])
    return df, label_encoders

def transform_with_label_encoders(df, label_encoders):
    for column, le in label_encoders.items():
        df[column] = df[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return df


def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != 'RainTomorrow':
            df[col] = df[col].astype(float)
    return df

import pandas as pd

# Filter extreme outliers
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

# Example usage
# filtered_df = drop_extreme_outliers(df)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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