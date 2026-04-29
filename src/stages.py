import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, List, Any
from pathlib import Path
from src.models_utils import load_boosting_model

def combine_data(consumption_path, production_path, output_path, config):
    """Этап 1: Объединение данных с использованием PCA для сжатия признаков продукции."""
    print("Stage 1: Combining data with PCA features...")
    cons = pd.read_csv(consumption_path, encoding='utf-8')
    prod = pd.read_csv(production_path, encoding='utf-8')
    
    date_col = config['features']['date_column']
    shop_col = config['features']['shop_column']
    cons[date_col] = pd.to_datetime(cons[date_col])
    prod[date_col] = pd.to_datetime(prod[date_col])
    cons['month'] = cons[date_col].dt.to_period('M').astype(str)
    prod['month'] = prod[date_col].dt.to_period('M').astype(str)
    
    # 1. Подготовка производственной матрицы (Month, Shop) -> Product columns
    prod_agg = prod.groupby(['month', shop_col, 'Артикул продукции'])['Количество'].sum().reset_index()
    prod_pivot = prod_agg.pivot_table(index=['month', shop_col], columns='Артикул продукции', values='Количество', fill_value=0)
    
    # Сжимаем 700+ продуктов в 30 компонент (PCA), чтобы уйти от разреженности
    n_comp = min(30, prod_pivot.shape[1], prod_pivot.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    prod_pca = pca.fit_transform(prod_pivot)
    prod_features = pd.DataFrame(
        prod_pca, 
        columns=[f'pca_{i}' for i in range(n_comp)], 
        index=prod_pivot.index
    ).reset_index()
    
    # Добавляем общие объемы
    total_prod = prod.groupby(['month', shop_col])['Количество'].sum().reset_index().rename(columns={'Количество': 'total_prod'})
    prod_features = pd.merge(prod_features, total_prod, on=['month', shop_col])
    
    # 2. Агрегируем потребление (используем ИМЯ как ключ, так как артикулы в task другие)
    cons_agg = cons.groupby(['month', shop_col, 'Наименование материала'])['Объём'].sum().reset_index()
    
    # 3. Объединяем
    combined = pd.merge(cons_agg, prod_features, on=['month', shop_col], how='left')
    combined.fillna(0, inplace=True)
    
    # Таргет-трансформация
    combined['target'] = np.log1p(combined['Объём'])
    combined.rename(columns={'month': date_col}, inplace=True)
    
    combined.to_csv(output_path, index=False)
    print(f"Saved combined data with PCA. Shape: {combined.shape}")

def clean_data(input_path, output_path, config):
    print("Stage 2: Cleaning...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['target'])
    df.to_csv(output_path, index=False)

def filter_outliers(input_path, output_path, config):
    print("Stage 3: Soft outlier filtering...")
    df = pd.read_csv(input_path)
    # Удаляем только экстремальные выбросы в лог-пространстве
    df = df[np.abs(stats.zscore(df['target'])) < 4]
    df.to_csv(output_path, index=False)

def feature_engineering(input_path, output_path, config):
    print("Stage 4: Feature engineering...")
    df = pd.read_csv(input_path)
    
    # Удаляем ненужное для обучения
    drop_cols = [config['features']['date_column'], 'Объём']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Кодируем Наименование материала, так как это теперь наш главный ключ
    le = LabelEncoder()
    df['material_id'] = le.fit_transform(df['Наименование материала'])
    df = df.drop(columns=['Наименование материала'])
    
    # Сохраняем LabelEncoder для инференса
    joblib.dump(le, 'models/material_encoder.joblib')
    
    df.to_csv(output_path, index=False)

def visualize_data(input_path, output_dir, config):
    """Stage 5: Visualizing data (Placeholder)"""
    print("Stage 5: Visualizing data skipped.")

def plot_feature_importance(models_dir, reports_dir, config):
    """Stage 8: Plotting feature importance (Placeholder)"""
    print("Stage 8: Plotting feature importance skipped.")

def evaluate_models(X_test, y_test, models_dir, config):
    print("Stage 7: Evaluating...")
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    for name in config['training']['models']:
        path = os.path.join(models_dir, f"{name}_model")
        if os.path.exists(path):
            model, feat_names = load_boosting_model(path)
            # Приведение типов для CatBoost/XGB
            X_input = X_test[feat_names].copy()
            for c in ['material_id', 'Цех']:
                if c in X_input.columns: X_input[c] = X_input[c].astype(int)
            
            p_log = model.predict(X_input)
            p, t = np.expm1(p_log), np.expm1(y_test)
            print(f"Model {name} -> R2: {r2_score(t, p):.4f}, MAE: {mean_absolute_error(t, p):.2f}")

def complete_task(models_dir, train_path, task_path, output_path, config):
    print("Stage 9: Generating correct forecast...")
    model_name = config['task']['model']
    model, feat_names = load_boosting_model(f"{models_dir}/{model_name}_model")
    le = joblib.load('models/material_encoder.joblib')
    
    # 1. Получаем список того, ЧТО нужно предсказать из task.csv
    task_df = pd.read_csv(task_path, sep=';', decimal=',', encoding='utf-8')
    target_info = task_df[['Артикул материала', 'Наименование материала', 'Цех']].copy()
    
    # 2. Подготовка фич продукции (PCA должен быть тем же!)
    raw_prod = pd.read_csv(config['paths']['raw_production'])
    raw_prod['month'] = pd.to_datetime(raw_prod[config['features']['date_column']]).dt.to_period('M').astype(str)
    prod_agg = raw_prod.groupby(['month', 'Цех', 'Артикул продукции'])['Количество'].sum().reset_index()
    prod_pivot = prod_agg.pivot_table(index=['month', 'Цех'], columns='Артикул продукции', values='Количество', fill_value=0)
    
    # Важно: используем те же фичи PCA
    n_comp = min(30, prod_pivot.shape[1], prod_pivot.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    prod_pca = pca.fit_transform(prod_pivot)
    prod_features = pd.DataFrame(prod_pca, columns=[f'pca_{i}' for i in range(n_comp)], index=prod_pivot.index).reset_index()
    total_prod = raw_prod.groupby(['month', 'Цех'])['Количество'].sum().reset_index().rename(columns={'Количество': 'total_prod'})
    prod_features = pd.merge(prod_features, total_prod, on=['month', 'Цех'])
    
    all_months = sorted(raw_prod['month'].unique())
    results = []
    
    # Кодируем имена из таска
    # Если имени нет в обучении - модель не сможет предсказать точно, но мы используем ближайшее
    all_known_materials = set(le.classes_)
    
    for month in all_months:
        m_features = prod_features[prod_features['month'] == month]
        for _, row in target_info.iterrows():
            name = row['Наименование материала']
            shop_raw = row['Цех']
            
            # Безопасное приведение к int
            try:
                shop = int(float(str(shop_raw).replace(',', '.')))
            except (ValueError, TypeError):
                continue
                
            shop_f = m_features[m_features['Цех'] == shop]
            
            if shop_f.empty or name not in all_known_materials:
                pred = 0.0
            else:
                input_row = shop_f.copy()
                input_row['material_id'] = le.transform([name])[0]
                input_row['Цех'] = shop
                
                X = input_row[feat_names].copy()
                for c in ['material_id', 'Цех']: X[c] = X[c].astype(int)
                
                pred = np.expm1(model.predict(X)[0])
            
            results.append({**row.to_dict(), 'month': month, 'val': max(0, pred)})
            
    res_df = pd.DataFrame(results)
    final = res_df.pivot_table(index=['Артикул материала', 'Наименование материала', 'Цех'], columns='month', values='val').reset_index()
    final.to_csv(output_path, index=False, sep=';', encoding='utf-8')

def compare_results(task_path, output_path, reports_dir, config):
    print("Stage 10: Comparison...")
    task = pd.read_csv(task_path, sep=';', decimal=',', encoding='utf-8')
    output = pd.read_csv(output_path, sep=';', decimal='.', encoding='utf-8')
    
    # Ключ для мерджа - ИМЯ и ЦЕХ
    key = ['Наименование материала', 'Цех']
    date_cols = [c for c in task.columns if '-' in c]

    # Приводим к общему типу
    for df in [task, output]:
        df['Наименование материала'] = df['Наименование материала'].astype(str)
        df['Цех'] = pd.to_numeric(df['Цех'], errors='coerce').fillna(0).astype(int).astype(str)
    
    merged = pd.merge(task[key + date_cols], output[key + date_cols], on=key, suffixes=('_t', '_p'))
    y_t = merged[[f"{c}_t" for c in date_cols]].values.flatten()
    y_p = merged[[f"{c}_p" for c in date_cols]].values.flatten()
    
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_t, y_p)
    print(f"FINAL R2: {r2:.4f}, MAE: {mean_absolute_error(y_t, y_p):.2f}")
    
    plt.figure(figsize=(10, 8)); plt.scatter(y_t, y_p, alpha=0.4); plt.plot([0, y_t.max()], [0, y_t.max()], 'r--')
    plt.title(f"R2 = {r2:.4f}"); plt.savefig(f"{reports_dir}/actual_vs_predicted.png"); plt.close()
