""" Usando a biblioteca Matplotlib, 
crie um gráfico circular (‘pie chart’) que represente a
média nos anos 2000 a 2015 da população total em milhões (“Population_mln”),
nos países Ghana, Kenya, Morocco e Nigeria. Coloque as legendas adequadas. """

import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV para um DataFrame do Pandas
df = pd.read_csv("../Life-Expectancy-Data-Updated.csv")

# Filtrar o DataFrame para os países e anos de interesse
countries = ["Ghana", "Kenya", "Morocco", "Nigeria"]
years = range(2000, 2016)

df_filtrado = df.loc[df.Country.isin(countries) & df.Year.isin(years)]

# Calcular a média da população total em milhões para cada país
pop_means = df_filtrado.groupby("Country")["Population_mln"].mean()

# Criar o gráfico circular
plt.pie(pop_means, labels=pop_means.index, autopct=lambda x: f"{x:.1f} M", )
plt.title("Média da População Total em Milhões (2000-2015)")

#Para modificar a legenda
plt.legend(title="Países", loc= (1, 0))
plt.show()