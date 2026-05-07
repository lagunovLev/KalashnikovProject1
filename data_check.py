import pandas as pd

df_cons = pd.read_csv('data/raw/consumption.csv')
df_cons['Дата'] = pd.to_datetime(df_cons['Дата'])

print("Минимальная дата в consumption.csv:", df_cons['Дата'].min())
print("Максимальная дата в consumption.csv:", df_cons['Дата'].max())
print("Всего уникальных месяцев в consumption.csv:", df_cons['Дата'].dt.to_period('M').nunique())