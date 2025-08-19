# Minimal, clean, project-ready training script from notebook
import os
import gzip
import pickle
import pickletools
import pandas as pd
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
# Assuming custom transformers are in this path
from src.features.custom_transformers import NumericTransformer, BinaryTransformer, MonthlyDataTransformer

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def save_model(filename: str, model: object):
    """Saves a model to a gzipped pickle file."""
    file_path = os.path.join(ROOT_DIR, "models", filename)
    with gzip.open(file_path, "wb") as f:
        pickled = pickle.dumps(model)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, model):
    """Creates a pipeline, trains it, and evaluates its performance."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate and print metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return pipeline

def tune_hyperparameters(X_train, y_train, preprocessor):
    """Tunes XGBoost hyperparameters using Optuna on a sample of the data."""
    # Use a smaller sample for faster tuning
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, train_size=0.5, random_state=57, stratify=y_train
    )

    def objective(trial):
        """Defines the objective function for Optuna to optimize."""
        param = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 3),
            'lambda': trial.suggest_float('lambda', 0, 3),
            'alpha': trial.suggest_float('alpha', 0, 3),
        }
        model = XGBClassifier(**param, random_state=57, use_label_encoder=False)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=57)
        f1_scores = cross_val_score(pipeline, X_train_sample, y_train_sample, cv=skf, scoring='f1')
        return f1_scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_trial.params

# --- Helper Functions for Preprocessing ---
def val_pattern(df):
    """Generates a list of column names that match the 'valX_Y' pattern."""
    arr = []
    for second in range(1, 7):
        for first in range(2, 22):
            s = f"val{first}_{second}"
            if s in df.columns:
                arr.append(s)
    return arr

def get_preprocessor(X_train):
    """Creates a ColumnTransformer to preprocess the data."""
    numeric_cols = ['age', 'tenure', 'age_dev', 'dev_num']
    binary_cols = ['is_dualsim', 'is_featurephone', 'is_smartphone']
    categorical_cols = ['trf', 'gndr', 'dev_man', 'device_os_name', 'simcard_type', 'region']
    
    # Find any other categorical columns that weren't explicitly listed
    extra_categoricals = [
        col for col in X_train.select_dtypes(include=['object', 'category', 'string']).columns 
        if col not in categorical_cols
    ]
    all_categorical_cols = categorical_cols + extra_categoricals
    
    monthly_cols = val_pattern(X_train)
    
    return ColumnTransformer(
        transformers=[
            ('numeric', NumericTransformer(), [col for col in numeric_cols if col in X_train.columns]),
            ('binary', BinaryTransformer(), [col for col in binary_cols if col in X_train.columns]),
            ('monthly', MonthlyDataTransformer(), [col for col in monthly_cols if col in X_train.columns]),
            ('categorical', CatBoostEncoder(), [col for col in all_categorical_cols if col in X_train.columns]),
        ],
        remainder='passthrough'
    )

def main():
    """Main function to run the training pipeline."""
    # Load data
    file_path = os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset.parquet")
    df = pd.read_parquet(file_path)
    
    target = 'target'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in processed data.")
    
    y = df[target]
    X = df.drop(columns=[target])
    

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57, stratify=y)

    # Save test set for prediction script
    external_dir = os.path.join(ROOT_DIR, "data", "external")
    os.makedirs(external_dir, exist_ok=True)
    X_test.to_parquet(os.path.join(external_dir, "X_test.parquet"))
    y_test.to_frame().to_parquet(os.path.join(external_dir, "y_test.parquet"))

    # Create preprocessor
    preprocessor = get_preprocessor(X_train)

    # --- Train and evaluate an initial baseline model ---
    print("--- Training initial model... ---")
    initial_model = XGBClassifier(random_state=57, eval_metric='logloss', use_label_encoder=False)
    train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, initial_model)

    # --- Tune hyperparameters ---
    print("\n--- Tuning hyperparameters... ---")
    best_params = tune_hyperparameters(X_train, y_train, preprocessor)
    print("Best parameters found:", best_params)

    # --- Train and evaluate the final model with best parameters ---
    print("\n--- Training final model... ---")
    final_model = XGBClassifier(**best_params, random_state=57, eval_metric='logloss', use_label_encoder=False)
    final_pipeline = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, final_model)
    
    # --- Save the final model pipeline ---
    print("\n--- Saving final model... ---")
    save_model("multisim_xgb.pkl.gz", final_pipeline)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
