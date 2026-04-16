import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

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

    # fix
    #cons = cons.groupby([date_col, shop_col]).sum(numeric_only=True).reset_index()
    #prod = prod.groupby([date_col, shop_col]).sum(numeric_only=True).reset_index()

    combined = pd.merge(prod, cons, on=[date_col, shop_col], suffixes=('_prod', '_cons'), how='inner')

    #if 'Количество_prod' in combined.columns:
    #    combined['Продукция'] = combined["Количество_prod"]
    #elif 'Количество' in combined.columns:
    #    combined['Продукция'] = combined['Количество']
#
    #if 'Объём_cons' in combined.columns:
    #    combined['Материал'] = combined["Объём_cons"]
    #elif 'Объём' in combined.columns:
    #    combined['Материал'] = combined['Объём']

    # fix 
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
    """Этап 4: Feature engineering + Target Encoding + Smart Encoding."""
    print("Stage 4: Feature engineering & Advanced Encoding...")
    df = pd.read_csv(input_path)
    target = config['features']['target_column']
    date_col = config['features']['date_column']
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Признаки даты
    #df['year'] = df[date_col].dt.year
    #df['month'] = df[date_col].dt.month
    #df['day'] = df[date_col].dt.day
    #df['day_of_week'] = df[date_col].dt.dayofweek
    #df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    #df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    #df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    #df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    #df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df = df.drop(columns=date_col)
    
    # 2. Target Encoding для Артикулов и Цехов
    # Мы создаем НОВЫЕ колонки, не удаляя старые
    #cols_for_target_enc = ['Артикул продукции', 'Артикул материала', 'Цех']
    #smoothing = 10 # Коэффициент сглаживания
    #global_mean = df[target].mean()
    
    #for col in cols_for_target_enc:
    #    if col in df.columns:
    #        print(f"Adding Target Encoding for {col}...")
    #        agg = df.groupby(col)[target].agg(['count', 'mean'])
    #        counts = agg['count']
    #        means = agg['mean']
            
    #        # Формула сглаженного среднего
    #        smooth_mean = (counts * means + smoothing * global_mean) / (counts + smoothing)
    #        df[f'{col}_target_enc'] = df[col].map(smooth_mean)
            
    # 3. Умное кодирование оригинальных текстовых колонок (OHE vs Label)
    # Чтобы модели могли работать с оригинальными данными в числовом виде
    #cols_to_encode = ['Цех', 'id документа_prod', 'id документа_cons', 'Артикул продукции', 'Артикул материала']
    cols_to_encode = ['Цех', 'Артикул продукции', 'Артикул материала']
    for col in cols_to_encode:
        if col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < 15:
                print(f"Applying One-Hot Encoding to {col} ({unique_count} unique values)")
                df = pd.get_dummies(df, columns=[col], prefix=col)
            else:
                print(f"Applying Label Encoding to {col} ({unique_count} unique values)")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    df.to_csv(output_path, index=False)
    print(f"Saved features data with Target Encoding to {output_path}")

def visualize_data(input_path, output_dir, config):
    """Этап 5: Визуализация."""
    print("Stage 5: Visualizing data...")
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(16, 14))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm') # annot=False для чистоты при куче колонок
    plt.title("Correlation Matrix (Target Encoding included)")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    
    target = config['features']['target_column']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); sns.boxplot(y=df[target]); plt.title(f"Boxplot: {target}")
    plt.subplot(1, 2, 2); sns.boxplot(y=df['Материал']); plt.title("Boxplot: Материал")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "boxplots.png")); plt.close()

    print(df.info())

def evaluate_models(X_test, y_test, models_dir, config):
    """Этап 7: Оценка."""
    print("Stage 7: Evaluating models...")
    import joblib
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    results = []
    models = {"XGBoost": ("xgboost_model.pkl", "joblib"), "LightGBM": ("lightgbm_model.pkl", "joblib"), "CatBoost": ("catboost_model.cbm", "catboost")}
    for name, (fname, mtype) in models.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path) if mtype == "joblib" else CatBoostRegressor().load_model(path)
            preds = model.predict(X_test)
            results.append({"Model": name, "RMSE": np.sqrt(mean_squared_error(y_test, preds)), "MAE": mean_absolute_error(y_test, preds), "R2": r2_score(y_test, preds)})
    res_df = pd.DataFrame(results)
    print("\nModel Metrics on Test Set:")
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(config['paths']['reports_dir'], "metrics.csv"), index=False)

def plot_feature_importance(models_dir, reports_dir, config):
    """Этап 8: Feature Importance."""
    print("Stage 8: Plotting feature importance...")
    import joblib
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    if os.path.exists(xgb_path):
        model = joblib.load(xgb_path)
        features = model.feature_names_in_
        importances = model.feature_importances_
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances, y=features)
        plt.title("XGBoost Feature Importance")
        plt.savefig(os.path.join(reports_dir, "feature_importance_xgb.png"))
        plt.close()

def complete_task(models_dir, task_path, output_path, config):
    """Этап 9: Выполнение задачи"""

