import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Le o ficheiro
life_expectancy = pd.read_csv("../Life-Expectancy-Data-Updated.csv")

# Seleciona apenas as colunas relevantes
life_expectancy = life_expectancy[['Year', 'GDP_per_capita', 'Polio', 'Life_expectancy', 'Country']]

# Seleciona só a região de Africa e o Gabão
life_expectancy = life_expectancy.loc[(life_expectancy.Country == 'Gabon')]

# Remove valores ausentes
life_expectancy.dropna(inplace=True)

# Separa os dados em treino e teste, deixando os 3 últimos anos para teste
X_train = life_expectancy[life_expectancy['Year'] <= 2015][['Year', 'Polio', 'GDP_per_capita']]
y_train_polio = life_expectancy[life_expectancy['Year'] <= 2015]['Polio']
y_train_gdp = life_expectancy[life_expectancy['Year'] <= 2015]['GDP_per_capita']

X_test = np.array([[2016], [2017], [2018]])
X_test = np.array([[2016, 0, 0], [2017, 0, 0], [2018, 0, 0]])


# Cria os modelos de regressão linear
model_polio = LinearRegression()
model_gdp = LinearRegression()

# Treina os modelos
model_polio.fit(X_train, y_train_polio)
model_gdp.fit(X_train, y_train_gdp)

# Faz as previsões para Polio e GDP_per_capita
X_test[:, 1] = model_polio.predict(X_test[:, [0, 1, 2]])
X_test[:, 2] = model_gdp.predict(X_test[:, [0, 1, 2]])


# Normaliza as variáveis
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cria o modelo de regressão linear para a esperança média de vida
y_train = life_expectancy[life_expectancy['Year'] <= 2015]['Life_expectancy']
model = LinearRegression()
model.fit(X_train, y_train)

# Faz as previsões para os anos de 2016, 2017 e 2018
X_pred = scaler.transform(X_test)
y_pred = model.predict(X_pred)

# Cria um gráfico para mostrar as previsões
years = [2016, 2017, 2018]
plt.plot(years, y_pred)
plt.title('Previsão da esperança média de vida para o Gabão')
plt.xlabel('Ano')
plt.ylabel('Esperança média de vida')
plt.show()

print("Previsões para a esperança média de vida:")
for year, prediction in zip(years, y_pred):
    print(f"{year}: {prediction:.2f}")
