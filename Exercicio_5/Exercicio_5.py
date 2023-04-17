import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../Exercicio_1/africa_data.csv')
df = df.dropna(subset=['GDP_per_capita', 'Life_expectancy'])

#So de dispersão por pais
sns.scatterplot(data=df, x="GDP_per_capita", y="Life_expectancy", hue="Country", 
                style="Country" )

# Labels
plt.xlabel('PIB per capita em dólares')
plt.ylabel('Esperança média de vida')
plt.title('Relação entre PIB per capita e Esperança Média de Vida')
plt.legend(title="Países", loc= (1.1, -1))
plt.show()

# Com linha de regressão global
g = sns.regplot(x="GDP_per_capita", y="Life_expectancy", data=df,
                scatter_kws = {"color": "black", "alpha": 0.5},
                line_kws = {"color": "red"},
                ci = 99)
# Labels
plt.xlabel('PIB per capita em dólares')
plt.ylabel('Esperança média de vida')
plt.title('Relação entre PIB per capita e Esperança Média de Vida')
plt.show()

