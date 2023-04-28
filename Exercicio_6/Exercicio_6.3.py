import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data from csv
df = pd.read_csv('../Exercicio_1/africa_data.csv')

# Filter data for Gabon only
gabon_df = df[df['Country'] == 'Gabon']

# Split data into features (X) and target (y)
X = gabon_df[['Year']]
y = gabon_df['Life_expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# Train a Linear Regression model on the training data
model = LinearRegression().fit(X_train, y_train)

# Predict Life Expectancy for 2020 and 2030
y_pred_2020 = model.predict([[2020]])[0]
y_pred_2030 = model.predict([[2030]])[0]

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics and predictions
print('MAE:', mae)
print('MSE:', mse)
print('R^2:', r2)
print('Esperança de vida no Gabão em 2020:', y_pred_2020)
print('Esperança de vida no Gabão em 2030:', y_pred_2030)

# Train a Linear Regression model on the entire dataset
model = LinearRegression().fit(X, y)

# Predict Life Expectancy for all years
y_pred = model.predict(X)

# Plot the Life Expectancy evolution over time
plt.plot(X, y, '-o', color='blue')
plt.plot(X, y_pred, color='red')
plt.plot([2020, 2030], [y_pred_2020, y_pred_2030], 'xr', color='green')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Evolução da esperança de vida no Gabão')
plt.legend(['Atual', 'Linha Regressão Linear', 'Valores previstos para 2020 e 2030'])
plt.xticks(range(2000, 2031, 5))
plt.xlim(2000, 2032)
plt.show()
