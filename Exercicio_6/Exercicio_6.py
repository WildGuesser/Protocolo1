import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar dados
df = pd.read_csv('../Exercicio_1/africa_data.csv')

df.drop(columns=['Economy_status_Developed', 'Economy_status_Developing'], inplace=True)

# Visualizar algumas informações do conjunto de dados
print(df.head())

# Verificar se existem valores nulos
print(df.isnull().sum())

# Separar as variáveis preditoras e a variável alvo
X = df[['Year', 'GDP_per_capita', 'Incidents_HIV']]
y = df['Life_expectancy']

# Normalizar os dados
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Country', 'Region'])), columns=df.columns[2:])
X_norm = df_norm[['Year', 'GDP_per_capita', 'Incidents_HIV']]
y_norm = df_norm['Life_expectancy']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R²:', r2_score(y_test, y_pred))

# Gerar mapa de calor da matriz de correlação
corr = df_norm.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

