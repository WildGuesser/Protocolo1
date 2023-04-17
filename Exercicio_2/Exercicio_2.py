""" 2. A partir do novo DataFrame, faça um gráfico que lhe permita visualizar
convenientemente a evolução das mortes de crianças menores de cinco anos por
1000 habitantes (“Under_five_deaths”) nos países Angola, Cabo Verde, Guinea e
Mozambique. """

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


#Carrega os dados
df = pd.read_csv('../Exercicio_1/africa_data.csv')

#Filtra para os países selecionados
paises = ['Angola', 'Cabo Verde', 'Guinea', 'Mozambique']
df_filtrado = df.loc[df.Country.isin(paises)]

#Cria o gráfico
sns.lineplot(data=df_filtrado, x='Year', y='Under_five_deaths', hue='Country')
plt.title('Mortes de Crianças Menores de 5 anos em Angola, Cabo Verde, Guiné e Moçambique', fontsize = 8)
plt.xlabel('Ano')
plt.ylabel('Mortes de Crianças Menores de 5 anos por 1000 habitantes', fontsize = 7)
plt.show()
