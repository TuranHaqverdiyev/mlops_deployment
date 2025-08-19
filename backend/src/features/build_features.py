import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def get_column_types(df):
    def val_pattern():
        arr = []
        for second in range(1, 7):
            for first in range(2, 22):
                s = f"val{first}_{second}"
                if s in df.columns:
                    arr.append(s)
        return arr

    numeric_cols = ['age', 'tenure', 'age_dev', 'dev_num']
    binary_cols = ['is_dualsim', 'is_featurephone', 'is_smartphone']
    categorical_cols = ['trf', 'gndr', 'dev_man', 'device_os_name', 'simcard_type', 'region']
        # Start with known categoricals
    extra_categoricals = [col for col in df.select_dtypes(include=['object', 'string']).columns if col not in categorical_cols]
    categorical_cols += extra_categoricals
    monthly_cols = val_pattern()
    return numeric_cols, binary_cols, categorical_cols, monthly_cols


class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, age_clip_min=18, age_clip_max=80):
        self.age_clip_min = age_clip_min
        self.age_clip_max = age_clip_max
        self.medians = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
            if col == 'age':
                X_copy[col] = X_copy[col].clip(self.age_clip_min, self.age_clip_max)
            self.medians[col] = X_copy[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
            if col == 'age':
                X_transformed[col] = X_transformed[col].clip(self.age_clip_min, self.age_clip_max)
            X_transformed[col] = X_transformed[col].fillna(self.medians[col])
        return X_transformed


class BinaryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
        return X_transformed


class MonthlyDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.medians[col] = X[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].fillna(self.medians[col])
        return X_transformed


def create_preprocessor(numeric_cols, binary_cols, categorical_cols, monthly_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', NumericTransformer(), numeric_cols),
            ('binary', BinaryTransformer(), binary_cols),
            ('monthly', MonthlyDataTransformer(), monthly_cols),
            ('categorical', CatBoostEncoder(cols=categorical_cols), categorical_cols),
        ],
        remainder='passthrough'
    )
    return preprocessor




# Minimal cleaning only, no target-aware encoding or fitting encoders
def preprocess(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df



def main():
    raw_path = os.path.join(ROOT_DIR, "data", "external", "multisim_dataset.parquet")
    processed_path = os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset.parquet")
    df = pd.read_parquet(raw_path)
    df_processed = preprocess(df)
    df_processed.to_parquet(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")


if __name__ == "__main__":
    main()
