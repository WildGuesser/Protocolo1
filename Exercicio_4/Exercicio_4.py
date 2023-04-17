"""
Crie uma função que, dado o nome do país da Região “Africa”, apresente o ano em
que a esperança média de vida (“Life_expectancy”) foi maior, bem como o
respetivo valor.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo CSV e carregá-lo em um DataFrame do Pandas
df = pd.read_csv('../Exercicio_1/africa_data.csv')

# Definir uma função que, dada o nome do país, apresente o ano em que a esperança média de vida foi a maior
def EMV(_country):
    # Filtrar o DataFrame para o país especificado
    df_country = df[df['Country'] == _country]
    
    # Encontrar a linha com o maior valor de esperança média de vida
    max_row = df_country.loc[df_country['Life_expectancy'].idxmax()]
    
    # Extrair o ano e o valor da esperança média de vida
    year = max_row['Year']
    life_expectancy = max_row['Life_expectancy']
    
    # Retornar o resultado como uma string
    return f"A maior esperança média de vida em {_country} foi de {life_expectancy:.1f} anos em {year}."

# Testar a função para o país "Gabão"
print(EMV("Gabon"))