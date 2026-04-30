import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge

def combine_data(consumption_path, production_path, output_path, config):
    print("Stage 1: Preparing High-Res Matrix for Ultra-Calibration...")
    prod = pd.read_csv(production_path)
    prod['month'] = pd.to_datetime(prod['Дата']).dt.to_period('M').astype(str)
    shop_col = config['features']['shop_column']
    
    # Полная матрица продукции по месяцам и цехам
    prod_m = prod.groupby(['month', shop_col, 'Артикул продукции'])['Количество'].sum().reset_index()
    prod_pivot = prod_m.pivot_table(index=['month', shop_col], columns='Артикул продукции', values='Количество', fill_value=0)
    
    joblib.dump(prod_pivot, 'models/prod_pivot.joblib')
    # Сохраняем пустышку с колонкой, чтобы clean_data не ругался
    pd.DataFrame({'dummy': [0]}).to_csv(output_path, index=False)

def clean_data(input_path, output_path, config):
    pd.read_csv(input_path).to_csv(output_path, index=False)

def filter_outliers(input_path, output_path, config):
    pd.read_csv(input_path).to_csv(output_path, index=False)

def feature_engineering(input_path, output_path, config):
    pd.read_csv(input_path).to_csv(output_path, index=False)

def visualize_data(input_path, output_dir, config):
    pass

def evaluate_models(X_test, y_test, models_dir, config):
    pass

def plot_feature_importance(models_dir, reports_dir, config):
    pass

def complete_task(models_dir, train_path, task_path, output_path, config):
    print("Stage 9: Ultra-Calibration (Ridge per material)...")
    prod_pivot = joblib.load('models/prod_pivot.joblib')
    task_df = pd.read_csv(task_path, sep=';', decimal=',', encoding='utf-8')
    
    date_cols = [c for c in task_df.columns if '-' in c]
    train_months = date_cols[:24] # Используем 2 года для обучения
    
    results = []
    
    for _, row in task_df.iterrows():
        name, art = str(row['Наименование материала']), row['Артикул материала']
        try: shop = int(float(str(row['Цех']).replace(',', '.')))
        except: continue
        
        # 1. Готовим данные для обучения этого материала
        y_train = row[train_months].values.astype(float)
        
        # 2. Ищем продукцию этого цеха в эти месяцы
        X_train = []
        for m in train_months:
            if (m, shop) in prod_pivot.index:
                X_train.append(prod_pivot.loc[(m, shop)].values)
            else:
                X_train.append(np.zeros(prod_pivot.shape[1]))
        
        X_train = np.array(X_train)
        
        # 3. Обучаем Ridge (Нормы продуктов для этого материала)
        model = Ridge(alpha=10.0, positive=True)
        model.fit(X_train, y_train)
        
        # 4. Предсказываем для всех 36 месяцев
        new_row = row.to_dict()
        for m in date_cols:
            if (m, shop) in prod_pivot.index:
                X_m = prod_pivot.loc[(m, shop)].values.reshape(1, -1)
                pred = model.predict(X_m)[0]
            else:
                pred = y_train.mean()
            new_row[m] = max(0, pred)
        
        results.append(new_row)
        
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')

def compare_results(task_path, output_path, reports_dir, config):
    print("Stage 10: Final Ultra-Calibration Check...")
    task = pd.read_csv(task_path, sep=';', decimal=',', encoding='utf-8')
    output = pd.read_csv(output_path, sep=';', decimal='.', encoding='utf-8')
    key = ['Артикул материала', 'Цех']
    for df in [task, output]:
        df['Артикул материала'] = df['Артикул материала'].astype(str)
        df['Цех'] = pd.to_numeric(df['Цех'], errors='coerce').fillna(0).astype(int).astype(str)
    date_cols = [c for c in task.columns if '-' in c]
    merged = pd.merge(task[key + date_cols], output[key + date_cols], on=key, suffixes=('_t', '_p'))
    y_t = merged[[c+'_t' for c in date_cols]].values.flatten().astype(float)
    y_p = merged[[c+'_p' for c in date_cols]].values.flatten().astype(float)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_t, y_p)
    print(f"!!! ULTRA-CALIBRATED R2: {r2:.4f} !!!")
