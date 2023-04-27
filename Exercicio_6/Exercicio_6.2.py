import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar dados
df = pd.read_csv('../Exercicio_1/africa_data.csv')

df.drop(columns=['Economy_status_Developed', 'Economy_status_Developing'], inplace=True)


# Normalizar os dados
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Country', 'Region'])), columns=df.columns[2:])


# Gerar mapa de calor da matriz de correlação
corr = df_norm.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()


# Através da leitura do mapa vamos selecionar apenas as colunas relevantes
life_expectancy = df[['Year', 'GDP_per_capita', 'Polio', 'Life_expectancy', 'Country']]

# Seleciona só o Gabão
life_expectancy = life_expectancy.loc[(life_expectancy.Country == 'Gabon')]

# Remove valores ausentes
life_expectancy.dropna(inplace=True)

# Seleciona as colunas de entrada e saída
X = life_expectancy[['Year', 'Polio', 'GDP_per_capita']]
y = life_expectancy['Life_expectancy']

# Normaliza as variáveis
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Cria o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Faz as previsões para os anos de 2000 a 2025
X_pred = np.array([[i, np.mean(life_expectancy.Polio), np.mean(life_expectancy.GDP_per_capita)] for i in range(2000, 2026)])
X_pred = scaler.transform(X_pred)
y_pred = model.predict(X_pred)

# Cria um gráfico para mostrar a evolução da esperança média de vida
years = list(range(2000, 2026))
plt.plot(years, y_pred, label='Previsão')
plt.scatter(life_expectancy.Year, life_expectancy.Life_expectancy, alpha=0.5, label='Dados observados')
plt.title('Evolução da esperança média de vida no Gabão')
plt.xlabel('Ano')
plt.ylabel('Esperança média de vida')
plt.legend()
plt.show()

# Faz as previsões para os anos de 2023, 2024 e 2025
X_pred = np.array([[2023, np.mean(life_expectancy.Polio), np.mean(life_expectancy.GDP_per_capita)],
                   [2024, np.mean(life_expectancy.Polio), np.mean(life_expectancy.GDP_per_capita)],
                   [2025, np.mean(life_expectancy.Polio), np.mean(life_expectancy.GDP_per_capita)]])
X_pred = scaler.transform(X_pred)
y_pred = model.predict(X_pred)

# Imprime as previsões
print(f"Previsão para 2023: {y_pred[0]}")
print(f"Previsão para 2024: {y_pred[1]}")
print(f"Previsão para 2025: {y_pred[2]}")
