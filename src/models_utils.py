import os
import joblib
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_data_splits(df, config):
    """Разделение данных на train/val/test."""
    target = config['features']['target_column']
    date_col = config['features']['date_column']
    
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
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            }
            model = XGBRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        joblib.dump({
          "model": best_model,
          "feature_names": X_train.columns.tolist()  # <-- вот они
          }, os.path.join(self.models_dir, "xgboost_model"))
        return best_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("Optimizing LightGBM...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'num_leaves': trial.suggest_int('num_leaves', 5, 150),
            }
            model = LGBMRegressor(**param, random_state=42, verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
        best_model.fit(X_train, y_train)
        joblib.dump({
          "model": best_model,
          "feature_names": X_train.columns.tolist()  # <-- вот они
          }, os.path.join(self.models_dir, "lightgbm_model"))
        return best_model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("Optimizing CatBoost...")
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 5000),
                'depth': trial.suggest_int('depth', 1, 16),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
            }
            model = CatBoostRegressor(**param, random_seed=42, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = CatBoostRegressor(**study.best_params, random_seed=42, verbose=False)
        best_model.fit(X_train, y_train)
        # Заменили нативный save_model на joblib для единообразия
        joblib.dump({
          "model": best_model,
          "feature_names": X_train.columns.tolist()  # <-- вот они
          }, os.path.join(self.models_dir, "catboost_model"))
        return best_model


def load_boosting_model(filepath):
    """
    Загружает модель, сохранённую через joblib.
    Возвращает готовый sklearn-совместимый объект.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл модели не найден: {filepath}")
    bundle = joblib.load(filepath)
    return bundle["model"], bundle["feature_names"]


def predict_from_dict(model, data_dict, feature_names):
    """
    Принимает joblib-модель, словарь с признаками и эталонный список имён признаков.
    Возвращает предсказание (float).
    """
    # 1. Проверка полноты данных
    missing = [f for f in feature_names if f not in data_dict]
    if missing:
        raise KeyError(f"В переданных данных отсутствуют признаки: {missing}")

    # 2. Формируем DataFrame в строгом порядке колонок (как при обучении)
    df_input = pd.DataFrame([data_dict])[feature_names]

    # 3. Предсказание
    pred = model.predict(df_input)

    # 4. Возвращаем скалярное значение
    return float(pred[0]) if hasattr(pred, '__len__') else float(pred)