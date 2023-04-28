import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Carregar dados
df = pd.read_csv('../Exercicio_1/africa_data.csv')

df.drop(columns=['Economy_status_Developed', 'Economy_status_Developing'], inplace=True)

df = df[df['Country'] == 'Gabon']

# Sort data by year
df = df.sort_values('Year')

# Normalizar os dados
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Country', 'Region'])), columns=df.columns[2:])


# Gerar mapa de calor da matriz de correlação
corr = df_norm.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Filtrar dados para ser so do Gabão
gabon_df = df[df['Country'] == 'Gabon']

# Selecionar features (X) e o target (y)
X = gabon_df[['Year', 'BMI', 'Schooling']]
y = gabon_df['Life_expectancy']

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# Treine um modelo de regressão linear nos dados de treino
model = LinearRegression().fit(X_train, y_train)

# Predict Life Expectancy for 2020 and 2030
X_2020 = [[2020, gabon_df['BMI'].iloc[-1], gabon_df['Schooling'].iloc[-1]]]
X_2030 = [[2030, gabon_df['BMI'].iloc[-1], gabon_df['Schooling'].iloc[-1]]]
y_pred_2020 = model.predict(X_2020)[0]
y_pred_2030 = model.predict(X_2030)[0]

# Avaliar o modelo no conjunto de teste
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir as métricas de avaliação e previsões
print('MAE:', mae)
print('MSE:', mse)
print('R^2:', r2)
print('Esperança de vida no Gabão em 2020:', y_pred_2020)
print('Esperança de vida no Gabão em 2030:', y_pred_2030)

# Treinar um modelo de regressão linear em todo o conjunto de dados
model = LinearRegression().fit(X, y)


# Prever a expectativa de vida para todos os anos
y_pred = model.predict(X)

# Traçar a evolução da expectativa de vida ao longo do tempo
plt.plot(X['Year'], y, '-o', color='blue')
plt.plot(X['Year'], y_pred, color='red')
plt.plot([2020, 2030], [y_pred_2020, y_pred_2030], 'xr', color='green')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Evolução da esperança de vida no Gabão')
plt.legend(['Atual', 'Linha Regressão Linear Múltipla', 'Valores previstos para 2020 e 2030'])
plt.xticks(range(2000, 2031, 5))
plt.xlim(2000, 2032)
plt.show()
