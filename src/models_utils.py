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
        try:
            if model_name == "xgboost":
                # XGBoost keys: validation_0, validation_1 ... metrics: rmse, mae ...
                train_key = 'validation_0'
                val_key = 'validation_1'
                metric = 'rmse' if 'rmse' in eval_results[train_key] else list(eval_results[train_key].keys())[0]
                train_rmse = eval_results[train_key][metric]
                val_rmse = eval_results[val_key][metric]
            elif model_name == "lightgbm":
                # LightGBM keys: training, valid_1 ... metrics: rmse, l2 ...
                train_key = 'training' if 'training' in eval_results else 'valid_0'
                val_key = 'valid_1'
                metric = 'rmse' if 'rmse' in eval_results[train_key] else 'l2' if 'l2' in eval_results[train_key] else list(eval_results[train_key].keys())[0]
                train_rmse = eval_results[train_key][metric]
                val_rmse = eval_results[val_key][metric]
            elif model_name == "catboost":
                train_rmse = eval_results['learn']['RMSE']
                val_rmse = eval_results['validation']['RMSE']
            
            plt.plot(train_rmse, label='Train')
            plt.plot(val_rmse, label='Validation')
            plt.title(f'Learning Curve: {model_name}')
            plt.xlabel('Iterations')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(self.reports_dir, f"learning_curve_{model_name}.png"))
        except Exception as e:
            print(f"Warning: Could not plot learning curve for {model_name}: {e}")
        finally:
            plt.close()

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("Deeply Optimizing XGBoost...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 16),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_bin': trial.suggest_int('max_bin', 256, 1024),
                'eval_metric': 'rmse',
                'n_jobs': -1
            }
            model = XGBRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        self._plot_learning_curve(best_model.evals_result(), "xgboost")
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, os.path.join(self.models_dir, "xgboost_model"))
        return best_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("Deeply Optimizing LightGBM...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 200),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'metric': 'rmse',
                'verbosity': -1,
                'n_jobs': -1
            }
            model = LGBMRegressor(**param, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.trials)
        
        best_model = LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)
        best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        self._plot_learning_curve(best_model.evals_result_, "lightgbm")
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, os.path.join(self.models_dir, "lightgbm_model"))
        return best_model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("Deeply Optimizing CatBoost...")
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 500, 4000),
                'depth': trial.suggest_int('depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 20.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'od_type': 'Iter',
                'od_wait': 100,
                'eval_metric': 'RMSE',
                'verbose': False,
                'thread_count': -1
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
        self._plot_learning_curve(best_model.get_evals_result(), "catboost")
        joblib.dump({"model": best_model, "feature_names": X_train.columns.tolist()}, os.path.join(self.models_dir, "catboost_model"))
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
