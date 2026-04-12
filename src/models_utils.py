import os
import joblib
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_data_splits(df, config):
    """Разделение данных на train/val/test."""
    target = config['features']['target_column']
    date_col = config['features']['date_column']
    
    # Теперь здесь нет автоматической фильтрации. 
    # Все колонки, которые есть в файле (кроме даты и таргета), пойдут в X.
    X = df.drop(columns=[target, date_col], errors='ignore')
    y = df[target]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['training']['test_size'] + config['training']['val_size'], 
        random_state=config['training']['random_state']
    )
    
    val_ratio = config['training']['val_size'] / (config['training']['test_size'] + config['training']['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, 
        random_state=config['training']['random_state']
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.trials = config['training']['optuna_trials']
        self.models_dir = config['paths']['models_dir']
        os.makedirs(self.models_dir, exist_ok=True)

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("Optimizing XGBoost...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = XGBRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, os.path.join(self.models_dir, "xgboost_model.pkl"))
        return best_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("Optimizing LightGBM...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            }
            model = LGBMRegressor(**param, random_state=42, verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = LGBMRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, os.path.join(self.models_dir, "lightgbm_model.pkl"))
        return best_model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("Optimizing CatBoost...")
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            }
            model = CatBoostRegressor(**param, random_seed=42, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = CatBoostRegressor(**study.best_params, random_seed=42, verbose=False)
        best_model.fit(X_train, y_train)
        best_model.save_model(os.path.join(self.models_dir, "catboost_model.cbm"))
        return best_model
