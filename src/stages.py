import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from src.models_utils import load_boosting_model
import joblib
from typing import Dict, List, Any
from pathlib import Path

def combine_data(consumption_path, production_path, output_path, config):
    """Этап 1: Объединение данных."""
    print("Stage 1: Combining data...")
    cons = pd.read_csv(consumption_path, encoding='utf-8')
    prod = pd.read_csv(production_path, encoding='utf-8')
    date_col = config['features']['date_column']
    shop_col = config['features']['shop_column']
    cons[date_col] = pd.to_datetime(cons[date_col])
    prod[date_col] = pd.to_datetime(prod[date_col])
    
    cols_to_remove = ['Единицы измерения', 'Наименование продукции', 'Наименование материала']
    cons = cons.drop(columns=[c for c in cols_to_remove if c in cons.columns])
    prod = prod.drop(columns=[c for c in cols_to_remove if c in prod.columns])

    combined = pd.merge(prod, cons, on=[date_col, shop_col], suffixes=('_prod', '_cons'), how='inner')
    combined = combined.sort_values(date_col)
    combined.rename(columns={'Количество': 'Продукция', 'Объём': 'Материал'}, inplace=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved combined data. Total rows: {len(combined)}")

def clean_data(input_path, output_path, config):
    """Этап 2: Очистка."""
    print("Stage 2: Cleaning data...")
    df = pd.read_csv(input_path)
    df.dropna(subset=[config['features']['target_column'], 'Материал'], inplace=True)
    df = df.drop(columns=config['features']['drop_columns'])
    df.to_csv(output_path, index=False)

def filter_outliers(input_path, output_path, config):
    """Этап 3: Удаление выбросов."""
    print("Stage 3: Filtering outliers...")
    df = pd.read_csv(input_path)
    target = config['features']['target_column']
    material = 'Материал'
    threshold = config['features']['outlier_threshold']
    df_filtered = df[(np.abs(stats.zscore(df[target])) < threshold)]
    df_final = df_filtered[(np.abs(stats.zscore(df_filtered[material])) < threshold)]
    df_final.to_csv(output_path, index=False)
    print(f"Removed {len(df) - len(df_final)} outlier rows.")

def feature_engineering(input_path, output_path, config):
    """Этап 4: Feature engineering + Encoding + Seasonality."""
    print("Stage 4: Feature engineering & Encoding & Seasonality...")
    import joblib
    df = pd.read_csv(input_path)
    target = config['features']['target_column']
    date_col = config['features']['date_column']
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Признаки сезонности
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['month', date_col])
    
    # 2. Кодирование категориальных признаков
    cols_to_encode = ['Цех', 'Артикул продукции', 'Артикул материала']
    encoders = {}
    
    for col in cols_to_encode:
        if col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < 15:
                print(f"Applying One-Hot Encoding to {col} ({unique_count} unique values)")
                categories = sorted(df[col].unique().tolist())
                encoders[col] = {'type': 'ohe', 'categories': categories}
                df = pd.get_dummies(df, columns=[col], prefix=col)
            else:
                print(f"Applying Label Encoding to {col} ({unique_count} unique values)")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': le}
    
    joblib.dump(encoders, os.path.join(config['paths']['models_dir'], "encoders.joblib"))
    df.to_csv(output_path, index=False)
    print(f"Saved features data and encoders to {output_path}")

def visualize_data(input_path, output_dir, config):
    """Этап 5: Визуализация."""
    print("Stage 5: Visualizing data...")
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Матрица корреляции
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(16, 14))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    
    # 2. Анализ целевой переменной
    target = config['features']['target_column']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[target])
    plt.title(f"Boxplot: {target}")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[target], kde=True)
    plt.title(f"Distribution: {target}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_analysis.png"))
    plt.close()

def evaluate_models(X_test, y_test, models_dir, config):
    """Этап 7: Оценка."""
    print("Stage 7: Evaluating models...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    results = []
    models_to_eval = {"XGBoost": "xgboost_model", "LightGBM": "lightgbm_model", "CatBoost": "catboost_model"}
    
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)

    for name, fname in models_to_eval.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model, _ = load_boosting_model(path)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, preds, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Fact')
            plt.ylabel('Pred')
            plt.title(f'Actual vs Predicted: {name} (R2={r2:.3f})')
            plt.savefig(os.path.join(reports_dir, f"actual_vs_predicted_{name.lower()}.png"))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            residuals = y_test - preds
            sns.histplot(residuals, kde=True)
            plt.title(f'Residuals Distribution: {name}')
            plt.savefig(os.path.join(reports_dir, f"residuals_{name.lower()}.png"))
            plt.close()

    res_df = pd.DataFrame(results)
    print("\nModel Metrics on Test Set:")
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(reports_dir, "metrics.csv"), index=False)

def plot_feature_importance(models_dir, reports_dir, config):
    """Этап 8: Feature Importance."""
    print("Stage 8: Plotting feature importance...")
    import joblib
    models = ["xgboost", "lightgbm", "catboost"]
    for m_name in models:
        path = os.path.join(models_dir, f"{m_name}_model")
        if os.path.exists(path):
            bundle = joblib.load(path)
            model = bundle['model']
            features = bundle['feature_names']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
                feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(20)
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=feat_imp)
                plt.title(f"{m_name.upper()} Top 20 Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(reports_dir, f"feature_importance_{m_name}.png"))
                plt.close()

def create_production_plan(production_path, output_path, config):
    """Этап 9: Создание плана производства (Таблица 1 Шага 2)."""
    print("Stage 9: Creating production plan from historical data...")
    df = pd.read_csv(production_path)
    date_col = config['features']['date_column']
    df[date_col] = pd.to_datetime(df[date_col])
    df['Month'] = df[date_col].dt.to_period('M').astype(str)
    
    pivot_df = df.pivot_table(
        index=['Артикул продукции', 'Наименование продукции', 'Цех'],
        columns='Month',
        values='Количество',
        aggfunc='sum'
    ).reset_index().fillna(0)
    
    pivot_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Production plan saved to {output_path}")

def complete_task(models_dir, train_path, plan_path, output_path, config):
    """Этап 10: Генерация прогноза потребления (оптимизировано)."""
    print("Stage 10: Generating consumption forecast (Batch processing)...")
    import joblib
    
    # 1. Загрузка маппинга
    df_train = pd.read_csv(train_path)
    mapping = df_train.groupby('Артикул продукции')['Артикул материала'].unique().to_dict()
    
    # 2. Загрузка модели и энкодеров
    model_name = config['task']['model']
    model, feature_names = load_boosting_model(os.path.join(models_dir, f"{model_name}_model"))
    encoders = joblib.load(os.path.join(models_dir, "encoders.joblib"))

    # 3. Подготовка данных плана
    plan_df = pd.read_csv(plan_path)
    time_cols = [c for c in plan_df.columns if c not in ['Артикул продукции', 'Наименование продукции', 'Цех', 'Единицы измерения']]
    
    long_plan = plan_df.melt(
        id_vars=['Артикул продукции', 'Цех'],
        value_vars=time_cols,
        var_name='Month',
        value_name='Продукция'
    )
    
    long_plan = long_plan[long_plan['Продукция'] > 0].copy()
    
    expanded_rows = []
    for _, row in long_plan.iterrows():
        prod_id = row['Артикул продукции']
        if prod_id in mapping:
            for mat_id in mapping[prod_id]:
                new_row = row.to_dict()
                new_row['Артикул материала'] = mat_id
                expanded_rows.append(new_row)
    
    if not expanded_rows:
        print("❌ No matches found for products in mapping.")
        return

    infer_df = pd.DataFrame(expanded_rows)
    
    # Добавляем признаки сезонности для инференса
    infer_df['dt'] = pd.to_datetime(infer_df['Month'])
    infer_df['month_num'] = infer_df['dt'].dt.month
    infer_df['month_sin'] = np.sin(2 * np.pi * infer_df['month_num'] / 12)
    infer_df['month_cos'] = np.cos(2 * np.pi * infer_df['month_num'] / 12)

    # Применяем энкодеры
    try:
        if 'Артикул продукции' in encoders:
            enc = encoders['Артикул продукции']
            infer_df['Артикул продукции_enc'] = enc['encoder'].transform(infer_df['Артикул продукции'].astype(str))
        
        if 'Артикул материала' in encoders:
            enc = encoders['Артикул материала']
            infer_df['Артикул материала_enc'] = enc['encoder'].transform(infer_df['Артикул материала'].astype(str))
            
        if 'Цех' in encoders:
            enc = encoders['Цех']
            if enc['type'] == 'ohe':
                for cat in enc['categories']:
                    infer_df[f'Цех_{cat}'] = (infer_df['Цех'].astype(str) == str(cat)).astype(float)

        # Подготовка X_infer
        X_infer = pd.DataFrame(index=infer_df.index)
        for feat in feature_names:
            if feat == 'Продукция':
                X_infer[feat] = infer_df['Продукция']
            elif feat == 'Артикул продукции':
                X_infer[feat] = infer_df['Артикул продукции_enc']
            elif feat == 'Артикул материала':
                X_infer[feat] = infer_df['Артикул материала_enc']
            elif feat in infer_df.columns:
                X_infer[feat] = infer_df[feat]
            else:
                X_infer[feat] = 0.0
        
        # Предсказание
        infer_df['Pred'] = np.maximum(0, model.predict(X_infer))
        
        output_pivot = infer_df.pivot_table(
            index=['Артикул материала', 'Цех'],
            columns='Month',
            values='Pred',
            aggfunc='sum'
        ).reset_index().fillna(0)
        
        for col in time_cols:
            if col not in output_pivot.columns:
                output_pivot[col] = 0.0
                
        output_pivot = output_pivot[['Артикул материала', 'Цех'] + time_cols]
        output_pivot.to_csv(output_path, index=False, encoding='utf-8', sep=';', decimal=',')
        print(f"✅ Forecast saved to {output_path}. Rows: {len(output_pivot)}")
        
    except Exception as e:
        print(f"❌ Error during batch inference: {e}")
