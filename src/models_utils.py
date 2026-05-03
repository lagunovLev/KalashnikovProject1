import os
import joblib
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_data_splits(df, config):
    """Разделение данных на train/val/test."""
    target = config['features']['target_column']
    
    X = df.drop(columns=[target], errors='ignore')
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
        self.reports_dir = config['paths']['reports_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _plot_learning_curve(self, eval_results, model_name):
        plt.figure(figsize=(10, 6))
        if model_name == "xgboost":
            train_rmse = eval_results['validation_0']['rmse']
            val_rmse = eval_results['validation_1']['rmse']
        elif model_name == "lightgbm":
            train_rmse = eval_results['training']['rmse']
            val_rmse = eval_results['valid_1']['rmse']
        elif model_name == "catboost":
            train_rmse = eval_results['learn']['RMSE']
            val_rmse = eval_results['validation']['RMSE']
        
        plt.plot(train_rmse, label='Train')
        plt.plot(val_rmse, label='Validation')
        plt.title(f'Learning Curve: {model_name}')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(os.path.join(self.reports_dir, f"learning_curve_{model_name}.png"))
        plt.close()

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("Optimizing XGBoost...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'eval_metric': 'rmse'
            }
            model = XGBRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_params = study.best_params
        best_params['eval_metric'] = 'rmse'
        best_model = XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        
        evals_result = best_model.evals_result()
        self._plot_learning_curve(evals_result, "xgboost")
        
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, 
                    os.path.join(self.models_dir, "xgboost_model"))
        return best_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("Optimizing LightGBM...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'metric': 'rmse',
                'verbosity': -1
            }
            model = LGBMRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_params = study.best_params
        best_params['metric'] = 'rmse'
        best_model = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
        best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        
        evals_result = best_model.evals_result_
        self._plot_learning_curve(evals_result, "lightgbm")
        
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, 
                    os.path.join(self.models_dir, "lightgbm_model"))
        return best_model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("Optimizing CatBoost...")
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 200, 2000),
                'depth': trial.suggest_int('depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'od_type': 'Iter',
                'od_wait': 50,
                'eval_metric': 'RMSE',
                'verbose': False
            }
            model = CatBoostRegressor(**param, random_seed=42)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_params = study.best_params
        best_params['verbose'] = False
        best_model = CatBoostRegressor(**best_params, random_seed=42)
        best_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        eval_metrics = best_model.get_evals_result()
        self._plot_learning_curve(eval_metrics, "catboost")
        
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, 
                    os.path.join(self.models_dir, "catboost_model"))
        return best_model


def load_boosting_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    bundle = joblib.load(filepath)
    return bundle["model"], bundle["feature_names"]


def predict_from_dict(model, data_dict, feature_names):
    missing = [f for f in feature_names if f not in data_dict]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    df_input = pd.DataFrame([data_dict])[feature_names]
    pred = model.predict(df_input)
    return float(pred[0]) if hasattr(pred, '__len__') else float(pred)
