from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
