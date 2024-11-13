import os
import sys
import time
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
import optuna
import datetime

def get_script_name():
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Load and preprocess training data
train_data = pd.read_csv("./input/train.csv").sample(frac=1.0, random_state=41)
X = train_data.drop(columns=["id", "Depression"])
y = train_data["Depression"]
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
X_num = X.select_dtypes(exclude=["object"])
X_cat = X.select_dtypes(include=["object"])
X_num_imputed = num_imputer.fit_transform(X_num)
X_cat_imputed = cat_imputer.fit_transform(X_cat)
X_imputed = pd.concat([
    pd.DataFrame(X_num_imputed, columns=X_num.columns),
    pd.DataFrame(X_cat_imputed, columns=X_cat.columns),
], axis=1)

# Generate polynomial features
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_imputed.select_dtypes(exclude=["object"]))
X_poly_df = pd.DataFrame(
    X_poly,
    columns=poly.get_feature_names_out(
        X_imputed.select_dtypes(exclude=["object"]).columns
    ),
)
X_final = pd.concat([X_poly_df, X_imputed[cat_features].reset_index(drop=True)], axis=1)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X_final.select_dtypes(exclude=["object"]).columns
X_final[numerical_cols] = scaler.fit_transform(X_final[numerical_cols])

# Convert categorical features to 'category' dtype
X_final[cat_features] = X_final[cat_features].astype('category')

# Define Objective Functions for Optuna
def objective_cat(trial):
    model = CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 550),
        depth=trial.suggest_int("depth", 4, 7),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        cat_features=cat_features,
        verbose=0,
        random_seed=41,
        early_stopping_rounds=50,
        task_type='CPU'  # Default to CPU; GPU handled separately
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    accuracy_list = []
    for train_index, valid_index in skf.split(X_final, y):
        X_train, X_valid = X_final.iloc[train_index], X_final.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, preds)
        accuracy_list.append(accuracy)
    return sum(accuracy_list) / len(accuracy_list)

def objective_xgb(trial):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 550),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=41,
        enable_categorical=True,
        verbosity=0
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    accuracy_list = []
    for train_index, valid_index in skf.split(X_final, y):
        X_train, X_valid = X_final.iloc[train_index], X_final.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, preds)
        accuracy_list.append(accuracy)
    return sum(accuracy_list) / len(accuracy_list)

# Initialize Optuna Studies
script_name = get_script_name()
storage_name = f'sqlite:///{script_name}.db'

study_cat = optuna.create_study(
    direction="maximize",
    study_name=f"{script_name}_cat",
    storage=storage_name,
    load_if_exists=True
)

study_xgb = optuna.create_study(
    direction="maximize",
    study_name=f"{script_name}_xgb",
    storage=storage_name,
    load_if_exists=True
)

# Determine whether to use GPU or CPU
ENABLE_STACKING = True
GPU = False
if os.path.exists('GPU'):
    GPU = True
elif os.path.exists('CPU'):
    GPU = False
else:
    GPU = False

print("Starting Optuna optimization loop. To stop, create a file named 'STOP' in the current directory.")

# Optimization Loop for CatBoost and XGBoost Only
while True:
    study_cat.optimize(objective_cat, n_trials=1)
    study_xgb.optimize(objective_xgb, n_trials=1)
    if os.path.exists('STOP'):
        print("STOP file detected. Exiting optimization loop.")
        break
    time.sleep(1)

# Function to Train and Validate Models
def train_and_validate(model, X, y, important_features, model_suffix):
    model.fit(X, y)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=41)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        break
    valid_preds = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, valid_preds)
    print(f"Validation Accuracy for {model_suffix}: {accuracy:.4f}")
    return model, accuracy

# Train and Validate CatBoost
if len(study_cat.trials) > 0:
    best_params_cat = study_cat.best_params
    print(f"Best hyperparameters for CatBoost: {best_params_cat}")
    best_model_cat = CatBoostClassifier(
        iterations=best_params_cat.get("iterations", 100),
        depth=best_params_cat.get("depth", 6),
        learning_rate=best_params_cat.get("learning_rate", 0.1),
        cat_features=cat_features,
        verbose=0,
        early_stopping_rounds=50,
        task_type='GPU' if GPU else 'CPU',
        random_seed=41
    )
    best_model_cat, accuracy_cat = train_and_validate(best_model_cat, X_final, y, None, "CatBoost")
    
    # Feature Importance and Selection
    feature_importances_cat = best_model_cat.get_feature_importance()
    important_features_cat = [
        feature
        for feature, importance in zip(X_final.columns, feature_importances_cat)
        if importance > 0
    ]
    print(f"Number of important features selected for CatBoost: {len(important_features_cat)}")
    X_final_cat = X_final[important_features_cat]
    cat_features_cat = [col for col in cat_features if col in important_features_cat]
    
    # Retrain CatBoost with Important Features
    best_model_cat = CatBoostClassifier(
        **best_params_cat,
        cat_features=cat_features_cat,
        verbose=0,
        early_stopping_rounds=50,
        task_type='GPU' if GPU else 'CPU',
        random_seed=41
    )
    best_model_cat.fit(X_final_cat, y)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=41)
    for train_index, valid_index in skf.split(X_final_cat, y):
        X_train, X_valid = X_final_cat.iloc[train_index], X_final_cat.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        break
    valid_preds_cat = best_model_cat.predict(X_valid)
    accuracy_cat = accuracy_score(y_valid, valid_preds_cat)
    print(f"Validation Accuracy for CatBoost: {accuracy_cat:.4f}")
else:
    print("No trials completed for CatBoost. Skipping CatBoost processing.")

# Train and Validate XGBoost
if len(study_xgb.trials) > 0:
    best_params_xgb = study_xgb.best_params
    print(f"Best hyperparameters for XGBoost: {best_params_xgb}")
    best_model_xgb = XGBClassifier(
        n_estimators=best_params_xgb.get("n_estimators", 100),
        max_depth=best_params_xgb.get("max_depth", 6),
        learning_rate=best_params_xgb.get("learning_rate", 0.1),
        subsample=best_params_xgb.get("subsample", 1.0),
        colsample_bytree=best_params_xgb.get("colsample_bytree", 1.0),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=41,
        enable_categorical=True,
        verbosity=0
    )
    best_model_xgb, accuracy_xgb = train_and_validate(best_model_xgb, X_final, y, None, "XGBoost")
    
    # Feature Importance and Selection
    feature_importances_xgb = best_model_xgb.feature_importances_
    important_features_xgb = [
        feature
        for feature, importance in zip(X_final.columns, feature_importances_xgb)
        if importance > 0
    ]
    print(f"Number of important features selected for XGBoost: {len(important_features_xgb)}")
    X_final_xgb = X_final[important_features_xgb]
    cat_features_xgb = [col for col in cat_features if col in important_features_xgb]
    
    # Retrain XGBoost with Important Features
    best_model_xgb = XGBClassifier(
        **best_params_xgb,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=41,
        enable_categorical=True,
        verbosity=0
    )
    best_model_xgb.fit(X_final_xgb, y)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=41)
    for train_index, valid_index in skf.split(X_final_xgb, y):
        X_train, X_valid = X_final_xgb.iloc[train_index], X_final_xgb.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        break
    valid_preds_xgb = best_model_xgb.predict(X_valid)
    accuracy_xgb = accuracy_score(y_valid, valid_preds_xgb)
    print(f"Validation Accuracy for XGBoost: {accuracy_xgb:.4f}")
else:
    print("No trials completed for XGBoost. Skipping XGBoost processing.")

# Optimize Stacking Classifier After Loop
if ENABLE_STACKING and len(study_cat.trials) > 0 and len(study_xgb.trials) > 0:
    # Initialize Optuna Study for Stacking Final Estimator
    study_sta = optuna.create_study(
        direction="maximize",
        study_name=f"{script_name}_sta",
        storage=storage_name,
        load_if_exists=True
    )
    
    # Define Objective Function for Stacking Final Estimator
    def objective_sta(trial):
        final_estimator_params = {
            'iterations': trial.suggest_int("iterations", 100, 550),
            'depth': trial.suggest_int("depth", 4, 10),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        }
        final_estimator = CatBoostClassifier(
            **final_estimator_params,
            verbose=0,
            random_seed=47,
            task_type='GPU' if GPU else 'CPU',
            thread_count=1 if GPU else -1,
            cat_features=[]  # No categorical features for final estimator
        )
        stacking_clf = StackingClassifier(
            estimators=[
                ("cat", best_model_cat),
                ("xgb", best_model_xgb),
            ],
            final_estimator=final_estimator,
            passthrough=False,
            n_jobs=1 if GPU else -1
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
        accuracy_list = []
        for train_index, valid_index in skf.split(X_final, y):
            X_train, X_valid = X_final.iloc[train_index], X_final.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            stacking_clf.fit(X_train, y_train)
            preds = stacking_clf.predict(X_valid)
            accuracy = accuracy_score(y_valid, preds)
            accuracy_list.append(accuracy)
        return sum(accuracy_list) / len(accuracy_list)
    
    print("Starting Optuna optimization for Stacking final estimator.")
    
    # Optimization Loop for Stacking Final Estimator
    while True:
        study_sta.optimize(objective_sta, n_trials=1)
        if os.path.exists('STOP'):
            print("STOP file detected during Stacking optimization. Exiting optimization loop.")
            break
        time.sleep(1)
    
    # Train and Validate Stacking Classifier
    if len(study_sta.trials) > 0:
        best_params_sta = study_sta.best_params
        print(f"Best hyperparameters for Stacking final estimator: {best_params_sta}")
        final_estimator_sta = CatBoostClassifier(
            **best_params_sta,
            verbose=0,
            random_seed=47,
            task_type='GPU' if GPU else 'CPU',
            thread_count=1 if GPU else -1,
            cat_features=[]  # No categorical features
        )
        stacking_clf = StackingClassifier(
            estimators=[
                ("cat", best_model_cat),
                ("xgb", best_model_xgb),
            ],
            final_estimator=final_estimator_sta,
            passthrough=False,
            n_jobs=1 if GPU else -1
        )
        stacking_clf, accuracy_sta = train_and_validate(stacking_clf, X_final, y, None, "Stacking")
    else:
        print("No trials completed for Stacking. Skipping Stacking processing.")
else:
    print("Stacking is not enabled or required models are missing.")

# Load and preprocess test data
test_data = pd.read_csv("./input/test.csv")
X_test = test_data.drop(columns=["id"])
X_test_num = X_test.select_dtypes(exclude=["object"])
X_test_cat = X_test.select_dtypes(include=["object"])
X_test_num_imputed = num_imputer.transform(X_test_num)
X_test_cat_imputed = cat_imputer.transform(X_test_cat)
X_test_imputed = pd.concat([
    pd.DataFrame(X_test_num_imputed, columns=X_num.columns),
    pd.DataFrame(X_test_cat_imputed, columns=X_cat.columns),
], axis=1)
X_test_poly = poly.transform(X_test_imputed.select_dtypes(exclude=["object"]))
X_test_poly_df = pd.DataFrame(
    X_test_poly,
    columns=poly.get_feature_names_out(
        X_test_imputed.select_dtypes(exclude=["object"]).columns
    ),
)
X_test_final = pd.concat([
    X_test_poly_df,
    X_test_imputed[cat_features].reset_index(drop=True)
], axis=1)
X_test_final[numerical_cols] = scaler.transform(X_test_final[numerical_cols])
X_test_final[cat_features] = X_test_final[cat_features].astype('category')

# Generate and Save Predictions

# Function to Save Predictions
def save_submission(predictions, model_suffix):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if model_suffix == "Stacking":
        submission_filename = f'submission-{timestamp}-{script_name}-sta-iNA-bNA-v{accuracy_sta:.4f}.csv'
    else:
        best_trial_number = getattr(study_cat.best_trial, 'number', 'NA') if model_suffix == "CatBoost" else getattr(study_xgb.best_trial, 'number', 'NA')
        best_trial_value = getattr(study_cat.best_trial, 'value', 'NA') if model_suffix == "CatBoost" else getattr(study_xgb.best_trial, 'value', 'NA')
        submission_filename = f'submission-{timestamp}-{script_name}-{model_suffix.lower()}-i{best_trial_number}-b{best_trial_value:.4f}-v{accuracy_cat if model_suffix == "CatBoost" else accuracy_xgb:.4f}.csv'
    
    submission = pd.DataFrame({
        "id": test_data["id"],
        "Depression": predictions
    })
    submission.to_csv(f"./working/{submission_filename}", index=False)
    print(f"Submission file for {model_suffix} saved to './working/{submission_filename}'.")

# CatBoost Predictions
if len(study_cat.trials) > 0:
    X_test_final_cat = X_test_final[important_features_cat]
    test_predictions_cat = best_model_cat.predict(X_test_final_cat)
    save_submission(test_predictions_cat, "CatBoost")

# XGBoost Predictions
if len(study_xgb.trials) > 0:
    X_test_final_xgb = X_test_final[important_features_xgb]
    test_predictions_xgb = best_model_xgb.predict(X_test_final_xgb)
    save_submission(test_predictions_xgb, "XGBoost")

# Stacking Predictions
if ENABLE_STACKING and len(study_cat.trials) > 0 and len(study_xgb.trials) > 0 and len(study_sta.trials) > 0:
    test_predictions_sta = stacking_clf.predict(X_test_final)
    save_submission(test_predictions_sta, "Stacking")

