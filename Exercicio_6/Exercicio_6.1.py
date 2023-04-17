# Definir valores para prever
X_futuro = pd.DataFrame({
    'Year': [2024, 2025, 2026, 2030],
    'GDP_per_capita': [1500, 1600, 1700, 2000],
    'Incidents_HIV': [5, 7, 9, 15]
})

# Normalizar os dados de X_futuro
X_futuro_norm = pd.DataFrame(scaler.transform(X_futuro), columns=X_futuro.columns)

# Fazer previsões para os anos de 2024, 2025, 2026 e 2030
y_futuro_norm = model.predict(X_futuro_norm)

# Desnormalizar as previsões
y_futuro = scaler.inverse_transform(y_futuro_norm.reshape(-1, 1)).flatten()

# Mostrar as previsões
print('Previsões para os anos de 2024, 2025, 2026 e 2030:')
print(y_futuro)

# Plotar gráfico de regressão linear para as previsões
X_completo = pd.concat([X_norm, X_futuro_norm])
y_completo = np.concatenate([y_norm, y_futuro_norm])

plt.figure(figsize=(10, 6))
sns.regplot(x='Year', y='Life_expectancy', data=df_norm, ci=None)
sns.regplot(x=X_completo['Year'], y=y_completo, ci=None, scatter=False)
plt.title('Regressão Linear para Previsão da Expectativa de Vida na África')
plt.xlabel('Ano')
plt.ylabel('Expectativa de Vida')
plt.show()
