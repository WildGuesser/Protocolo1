""" 1. Carregue o ficheiro .csv para um DataFrame, e de seguida crie um novo
DataFrame com apenas a informação da Região “Africa”. Grave este novo
DataFrame num novo ficheiro .csv."""

import pandas as pd

# Le o ficheiro
life_expectancy = pd.read_csv("../Life-Expectancy-Data-Updated.csv")

#Seleciona só a região de Africa
life_expectancy = life_expectancy.loc[life_expectancy.Region == 'Africa']

#Escreve um novo ficheiro com a nova informação
life_expectancy.to_csv("africa_data.csv", index=False)