import pandas as pd
import re
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from geopy.distance import great_circle
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import math
from sklearn.neighbors import NearestNeighbors
import heapq
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

dados = pd.read_excel('drive/MyDrive/Colab Notebooks/total_mod.xlsx')

df = dados.copy() #dados finais apos o processo de limpeza
df = df.dropna(how='all')# Remove linhas completamente vazias
df.drop(["review_scores_cleanliness", "review_scores_checkin", "review_scores_communication"], axis=1, inplace=True)#faça o drop dessas colunas
print(df.isna().sum())

df.columns

#tranforma todos valores possiveis em numerico
df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
df.head()

#corrigir colunas que tem "$" e ","
for i in range(11, 18):
    if i != 16:
        coluna = df.columns[i]
        df[coluna] = df[coluna].apply(lambda x: float(str(x).replace('$', '').replace(',', '')))

df.head()

#faz as colunas dos tipos de propriedade ficar numerico
def tipo_number(df, coluna):
    #converte n strings em tipos distintos
    valor_unico = df[coluna].unique()
    tipos = {value: i for i, value in enumerate(valor_unico)}
    df[coluna] = df[coluna].map(tipos)
    tipos=0
    valor_unioc=0

tipo_number(df, df.columns[2])
tipo_number(df, df.columns[3])
tipo_number(df, df.columns[8])
df.head()

# Função para limpar e formatar os valores
def trunca(valor):
    valor = str(valor).replace(',', '').replace('.', '').replace('e+', '').replace('e-', '')  # Converter para string
    valor = valor[:10]  # Truncar para 10 dígitos
    valor = valor[:3] + '.' + valor[3:]  # Corrigir para latitude e longitude
    return valor

df_copy = df.copy()

# Elimine os itens com valores None
df_copy.dropna(subset=['latitude', 'longitude'], inplace=True)

df_copy.iloc[:, 0] = df_copy.iloc[:, 0].apply(trunca).astype(float)
df_copy.iloc[:, 1] = df_copy.iloc[:, 1].apply(trunca).astype(float)

df = df_copy.copy()
df_copy = None
df.head()

from sklearn.neighbors import KNeighborsRegressor

df_copy = df.copy()

df_missing = df_copy[df_copy['review_scores_location'].isna()]
df_not_missing = df_copy[~df_copy['review_scores_location'].isna()]

# Criar um modelo k-NN
knn_model = KNeighborsRegressor(n_neighbors=2)  # 2 vizinhos é o melhor pela validação cruzada

knn_model.fit(df_not_missing[['latitude', 'longitude']], df_not_missing['review_scores_location'])

imputed_scores = knn_model.predict(df_missing[['latitude', 'longitude']])

df_copy.loc[df_copy['review_scores_location'].isna(), 'review_scores_location'] = imputed_scores

#calcular precisão
if False:
  X_train, X_test, y_train, y_test = train_test_split(df_copy[['latitude', 'longitude']], df_copy['review_scores_location'], test_size=0.2, random_state=42)

  imputed_test_scores = knn_model.predict(X_test)

  mse = mean_squared_error(y_test, imputed_test_scores)

  r2 = r2_score(y_test, imputed_test_scores)

  print(f'Mean Squared Error (MSE): {mse}')
  print(f'R-squared (R²): {r2}')

df = df_copy.copy()
del df_copy
df

if False:
  # Criar o gráfico de dispersão
  plot_df = df[['latitude', 'longitude', 'review_scores_location']]

  colors = ['#8B0000', '#A52A2A', '#B22222', '#CD5C5C', '#FF4500', '#FF6347', '#FF7F50', '#FFA07A', '#98FB98', '#008000']
  cmap = ListedColormap(colors)

  plt.figure(figsize=(10, 6))
  scatter = plt.scatter(plot_df['longitude'], plot_df['latitude'], c=plot_df['review_scores_location'], cmap=cmap, s=10)
  plt.colorbar(scatter, label='review_scores_location')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('review_scores_location das coordenadas')
  plt.show()

print(df["amenities"].iloc[0])

#criar um mapa chave item -> e um pair com vetor de precos e quantidade
comodidades_dict = {}

for index, row in df.iterrows():
    comodidades = row['amenities']  
    price = row['price'] 
    
    #dividir a lista
    comodidades_list = comodidades.split(',')
    
    #iterar sobre as comodidades individuais
    for comodidade in comodidades_list:
        comodidade = comodidade.strip()  # Remova espaços em branco em excesso
        comodidade = comodidade.replace('}', '').replace('{', '').replace('-', '').replace('/', '')
        comodidade = comodidade.lower()
        
        if any(c.isalpha() for c in comodidade):
            if comodidade in comodidades_dict:
                # Se a comodidade já estiver no dicionário, atualize o vetor de preços e o contador
                comodidades_dict[comodidade][0].append(price)
                comodidades_dict[comodidade][1] += 1
            else:
                # Se a comodidade não estiver no dicionário, crie uma entrada para ela.
                comodidades_dict[comodidade] = [[price], 1]

map_preco_normalizado = {}  
print(len(comodidades_dict)) 

min_preco = min(preco for precos, count in comodidades_dict.values() for preco in precos)
max_preco = max(preco for precos, count in comodidades_dict.values() for preco in precos)

for comodidade, (precos, count) in comodidades_dict.items():
    preco_normalizado = (sum(precos) / count - min_preco) / (max_preco - min_preco)
    map_preco_normalizado[comodidade] = preco_normalizado
    print(comodidade, preco_normalizado)

df_copy = df.copy()

for index, row in df.iterrows():
    comodidades = row['amenities']
    comodidades_list = comodidades.split(',')
    total_rating = 0

    for comodidade in comodidades_list:
        comodidade = comodidade.strip()
        comodidade = comodidade.replace('}', '').replace('{', '').replace('-', '').replace('/', '')
        comodidade = comodidade.lower()

        if any(c.isalpha() for c in comodidade) and comodidade in map_preco_normalizado:
            total_rating += map_preco_normalizado[comodidade]

    # Armazena o rating total no DataFrame copiado
    df_copy.at[index, 'rating_amenities'] = total_rating


df_copy
