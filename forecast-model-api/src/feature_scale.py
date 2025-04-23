import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

us_holidays = holidays.UnitedStates()

def add_time_features(df):
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['is_holiday'] = df['date'].isin(us_holidays).astype(int)
    
    # Fourier transforms
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    return df

def add_lag_features(df, lags=[7, 14, 30]):
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    return df

def add_rolling_features(df, window=30):
    df[f'sales_roll_mean_{window}'] = df.groupby(['store', 'item'])['sales'].shift(1).rolling(window).mean()
    return df

def target_encode(df, col, target='sales'):
    means = df.groupby(col)[target].mean()
    df[f'{col}_encoded'] = df[col].map(means)
    return df

def generate_feature(df, history_df):
    df = df.merge(history_df, on=['store', 'item'], how='left')

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = target_encode(df, 'store')
    df = target_encode(df, 'item')
    
    # Drop unused
    drop_cols = ['sales', 'store', 'item', 'date']
    df = df.drop(columns=[col for col in drop_cols if col in df])
    
    return df

def feature_scale(inputdata, history_df, scaler_path="./Model/scaler.pkl"):
    df = pd.DataFrame([inputdata])
    df['date'] = pd.to_datetime(df['date'])
    
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    features = generate_feature(df, history_df)
    features.fillna(0, inplace=True)  # Handle NaNs
    
    # Load scaler if needed
    import joblib
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)
    
    return features_scaled
