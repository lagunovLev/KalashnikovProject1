import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from src.models_utils import load_boosting_model
import pandas as pd
import numpy as np
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



def predict_from_dict(model, data_dict, feature_names, model_type='sklearn_like'):
    """
    Принимает модель, словарь с признаками и список имён признаков.
    Возвращает предсказание для одной строки данных.
    """
    # 1. Проверяем, что все нужные признаки присутствуют в словаре
    missing = [f for f in feature_names if f not in data_dict]
    if missing:
        raise KeyError(f"В переданных данных отсутствуют признаки: {missing}")

    # 2. Преобразуем словарь в DataFrame с СТРОГИМ порядком столбцов
    # Это обязательно, иначе модель перепутает признаки и предсказание будет неверным
    df_input = pd.DataFrame([data_dict])[feature_names]

    # 3. Делаем предсказание в зависимости от типа модели
    if model_type == 'xgboost':
        pred = model.predict(df_input)
    elif model_type == 'catboost':
        pred = model.predict(df_input)
    elif model_type == 'lightgbm':
        pred = model.predict(df_input)
    else:
        # Для sklearn-совместимых моделей, сохранённых через joblib/pickle
        pred = model.predict(df_input)

    # Возвращаем скалярное значение (убираем обёртку numpy array)
    return float(pred[0]) if hasattr(pred[0], 'item') else pred[0]


def calculate_materials(
    task_path: str,
    model,
    feature_names: List[str],
    product_to_materials: Dict[int, List[int]],
    product_col: str = 'артикул продукции',
    workshop_col: str = 'цех',
    material_col: str = 'артикул материала'
):
    """
    Преобразует план производства продукции в план потребности материалов.
    
    Args:
        task_path: Путь к входной таблице (продукция × цех × периоды)
        output_path: Путь для сохранения результата (материалы × цех × периоды)
        model: Обученная модель для предсказания количества
        feature_names: Список признаков модели
        product_to_materials: Словарь {артикул_продукции: [артикул_материала1, ...]}
        product_col: Название колонки с продукцией во входном файле
        workshop_col: Название колонки с цехом
        material_col: Название колонки для материалов в выходном файле
    """
    # 1. Загрузка данных
    df = pd.read_csv(task_path, encoding="1125")
    
    # 2. Определение колонок времени (всё кроме продукции и цеха)
    time_cols = [c for c in df.columns if c not in [product_col, workshop_col]]
    
    if not time_cols:
        raise ValueError("Не найдены колонки с периодами времени")
    
    # 3. Сбор всех результатов предсказаний
    results = []
    
    for idx, row in df.iterrows():
        product = row[product_col]
        workshop = row[workshop_col]
        
        # Проверка наличия продукта в словаре материалов
        if product not in product_to_materials:
            print(f"⚠️ Продукт {product} не найден в словаре материалов, пропускаем")
            continue
        
        materials = product_to_materials[product]
        
        # Предсказание для каждого материала
        for material in materials:
            # Формируем входной словарь для модели
            data_dict = {
                product_col: product,
                workshop_col: workshop,
                **{col: row[col] for col in time_cols}
            }
            
            # Добавляем материал как признак (если он есть в feature_names)
            if material_col in feature_names:
                data_dict[material_col] = material
            
            try:
                quantity = predict_from_dict(model, data_dict, feature_names)
                
                results.append({
                    material_col: material,
                    workshop_col: workshop,
                    **{col: quantity for col in time_cols}
                })
            except Exception as e:
                print(f"⚠️ Ошибка предсказания для {product}/{material}/{workshop}: {e}")
                continue
    
    # 4. Создание выходной таблицы
    if not results:
        raise ValueError("Не удалось сделать ни одного предсказания")
    
    df_output = pd.DataFrame(results)
    
    # 5. Агрегация (если один материал встречается несколько раз)
    agg_dict = {col: 'sum' for col in time_cols}
    agg_dict[workshop_col] = 'first'
    df_output = df_output.groupby(material_col, as_index=False).agg(agg_dict)
    
    print(f"✅ Готово! Сохранено {len(df_output)} строк в {output_path}")
    return df_output


def complete_task(models_dir, train_path, task_path, output_path, config):
    """Этап 9: Выполнение задачи"""
    import csv
    from collections import defaultdict
    

    def csv_to_dict(filepath, col_a='Артикул продукции', col_b='Артикул материала'):
        """
        Читает CSV и возвращает словарь: 
        ключ = значения колонки A, значение = список соответствующих значений B.
        """
        grouped = defaultdict(list)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = int(row.get(col_a, '').strip())
                value = int(row.get(col_b, '').strip())

                if key:  # Пропускаем строки, где колонка A пустая
                    grouped[key].append(value)

        return dict(grouped)
    
    material_articles = csv_to_dict(train_path)
    model, feature_names = load_boosting_model(f"{models_dir}/{config["task"]["model"]}_model")

    output = calculate_materials(task_path, model, feature_names, material_articles, "Артикул продукции", "Цех", "Артикул материала")
    output.to_csv(output_path, index=False, encoding='utf-8')

    

