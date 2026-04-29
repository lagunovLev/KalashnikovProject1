import os
import yaml
import argparse
import pandas as pd
from src.cache_utils import PipelineState
from src.stages import (
    combine_data, clean_data, filter_outliers, 
    feature_engineering, visualize_data, 
    evaluate_models, plot_feature_importance,
    complete_task, compare_results
)
from src.models_utils import ModelTrainer, get_data_splits
import io

def load_config(config_path="config.yaml"):
    with io.open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_pipeline(args):
    config = load_config()
    state = PipelineState(config['paths']['state_file'])
    paths = config['paths']
    
    # --- Stage 1: Combine ---
    if args.stage <= 1:
        if state.is_step_needed("combine", combine_data, [paths['raw_consumption'], paths['raw_production']], config['csv_params']):
            combine_data(paths['raw_consumption'], paths['raw_production'], paths['combined_data'], config)
            state.update_step("combine", combine_data, [paths['raw_consumption'], paths['raw_production']], config['csv_params'])
        else:
            print("Stage 1 (Combine) skipped: no changes.")

    # --- Stage 2: Clean ---
    if args.stage <= 2:
        if state.is_step_needed("clean", clean_data, [paths['combined_data']], config['features']):
            clean_data(paths['combined_data'], paths['clear_data'], config)
            state.update_step("clean", clean_data, [paths['combined_data']], config['features'])
        else:
            print("Stage 2 (Clean) skipped: no changes.")

    # --- Stage 3: Outliers ---
    if args.stage <= 3:
        if state.is_step_needed("outliers", filter_outliers, [paths['clear_data']], config['features']):
            filter_outliers(paths['clear_data'], paths['filtered_data'], config)
            state.update_step("outliers", filter_outliers, [paths['clear_data']], config['features'])
        else:
            print("Stage 3 (Outliers) skipped: no changes.")

    # --- Stage 4: Feature Engineering ---
    if args.stage <= 4:
        if state.is_step_needed("features", feature_engineering, [paths['filtered_data']], config['features']):
            feature_engineering(paths['filtered_data'], paths['features_data'], config)
            state.update_step("features", feature_engineering, [paths['filtered_data']], config['features'])
        else:
            print("Stage 4 (Features) skipped: no changes.")

    # --- Stage 5: Visualization ---
    if args.stage <= 5:
        visualize_data(paths['features_data'], paths['reports_dir'], config)

    # --- Stage 6: Training ---
    if args.stage <= 6:
        print("Stage 6: Training models...")
        df = pd.read_csv(paths['features_data'])
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(df, config)
        
        trainer = ModelTrainer(config)
        
        if args.model in ["all", "xgboost"]:
            trainer.train_xgboost(X_train, y_train, X_val, y_val)
        if args.model in ["all", "lightgbm"]:
            trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        if args.model in ["all", "catboost"]:
            trainer.train_catboost(X_train, y_train, X_val, y_val)

    # --- Stage 7: Evaluation ---
    if args.stage <= 7:
        df = pd.read_csv(paths['features_data'])
        _, _, X_test, _, _, y_test = get_data_splits(df, config)
        evaluate_models(X_test, y_test, paths['models_dir'], config)

    # --- Stage 8: Plots ---
    #if args.stage <= 8:
    #    plot_feature_importance(paths['models_dir'], paths['reports_dir'], config)

    if args.stage <= 9:
        complete_task(paths['models_dir'], paths['features_data'], paths['task'], paths['task_output'], config)

    if args.stage <= 10:
        compare_results(paths['task'], paths['task_output'], paths['reports_dir'], config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalashnikov ML Pipeline")
    parser.add_argument("--stage", type=int, default=1, help="Start from this stage (1-8)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "xgboost", "lightgbm", "catboost"], help="Model to train")
    
    args = parser.parse_args()
    run_pipeline(args)
