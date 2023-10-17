import pandas as pd
import re
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from geopy.distance import great_circle
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
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
import itertools
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

dados = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/total_mod.xlsx')

print(len(dados))

df = dados.copy() #dados finais apos o processo de limpeza
df = df.dropna(how='all')# Remove linhas completamente vazias
df.drop(["review_scores_cleanliness", 'review_scores_value','review_scores_rating', 'review_scores_accuracy', "review_scores_checkin", "review_scores_communication","availability_30","availability_60","availability_90","availability_365","extra_people","security_deposit",'weekly_price', 'monthly_price','cleaning_fee','square_feet'], axis=1, inplace=True)#faça o drop dessas colunas
print(df.isna().sum())

df.dtypes

df['price'].head()

#corrigir colunas que tem "$" e ","
df['price'] = df['price'].apply(lambda x: float(str(x).replace('$', '').replace(',', '')))

df['price'].head()

#retira valores muito grandes que tem pouca quantidade
df_copy = df[df['price'] < 60000]

n, bins, patches = plt.hist(df_copy['price'], 50, density=True, facecolor='g', alpha=0.75, log = True)
plt.title("Pricing Distribution")
plt.xlabel("Price $")
plt.ylabel("Counts (log)")

stopwords = set(STOPWORDS)
words = df[['property_type']].copy()
words['property_type'] = words['property_type'].apply(lambda x: x.strip())
counts = words.property_type.value_counts()

wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=50,
                          max_font_size=100
).generate_from_frequencies(counts)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Group the data frame by property type and extract a number of stats from each group
df_copy = df[df['price'] < 60000]
df_copy.groupby(['property_type']).agg({
    # find the min, max, and sum of the price column
    'price': ['min', 'max', 'mean', 'count']})

#faz as colunas dos tipos de propriedade ficar numerico
def tipo_number(df,coluna):
    #converte n strings em tipos distintos
    valor_unico = df[coluna].unique()
    tipos = {value: i for i, value in enumerate(valor_unico)}
    df[coluna] = df[coluna].map(tipos)
    tipos=0
    valor_unioc=0

tipo_number(df,'property_type')
tipo_number(df,'room_type')
tipo_number(df,'bed_type')

df[['property_type', 'room_type', 'bed_type']].head()

#1- Fazer os calculos de review_scores_location com base na localização

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
df[['latitude', 'longitude']].head()


from sklearn.neighbors import KNeighborsRegressor

df_copy = df.copy()

df_missing = df_copy[df_copy['review_scores_location'].isna()]
df_not_missing = df_copy[~df_copy['review_scores_location'].isna()]

# Criar um modelo k-NN
knn_model = KNeighborsRegressor(n_neighbors=2)  # 2 vizinhos é o melhor pela validação cruzada

knn_model.fit(df_not_missing[['latitude', 'longitude']], df_not_missing['review_scores_location'])

imputed_scores = knn_model.predict(df_missing[['latitude', 'longitude']])

df_copy.loc[df_copy['review_scores_location'].isna(), 'review_scores_location'] = imputed_scores


df = df_copy.copy()
del df_copy
df['review_scores_location'].head()

if True:
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

#2 - Corrigir dados do amenities

results = Counter()
df_aux = df.copy()
df_aux['amenities'].str.strip('{}')\
               .str.replace('"', '')\
               .str.lstrip('\"')\
               .str.rstrip('\"')\
               .str.split(',')\
               .apply(results.update)

#results.most_common(20)
# create a new dataframe
am_df = pd.DataFrame(results.most_common(20),
                     columns=['amenity', 'count'])

# plot the Top 20
am_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',
                                                      figsize=(10,7), legend=False,
                                                      width=1.0,align="center",
                                                      title='Amenities')
plt.xlabel('Count');

print(df["amenities"].iloc[0])

sinonimo = {
    "wifi": "internet",
}

#criar um mapa chave item -> e um pair com vetor de precos e quantidade
comodidades_dict = {}

for index, row in df.iterrows():
    comodidades = row['amenities']
    price = row['price']

    comodidades_list = comodidades.split(',')

    for comodidade in comodidades_list:
        comodidade = comodidade.strip()  # Remova espaços em branco em excesso
        comodidade = comodidade.replace(' ', '').replace('}', '').replace('{', '').replace('-', '').replace('/', '').replace('"', '').replace("'", '')
        comodidade = comodidade.lower()

        if any(c.isalpha() for c in comodidade) and "translationmissing" not in comodidade:
            #comodidade = sinonimo.get(comodidade, comodidade)
            if comodidade in comodidades_dict:
                # Se a comodidade já estiver no dicionário, atualize o vetor de preços e o contador
                comodidades_dict[comodidade][0].append(price)
                comodidades_dict[comodidade][1] += 1
            else:
                # Se a comodidade não estiver no dicionário, crie uma entrada para ela.
                comodidades_dict[comodidade] = [[price], 1]

lista_de_chaves = list(comodidades_dict.keys())

print(lista_de_chaves)

from sklearn.preprocessing import MinMaxScaler

# Processar os dados em comodidades_dict para calcular os preços médios
comodidades = []
precos = []

for comodidade, (precos_raw, quantidade) in comodidades_dict.items():
    comodidades.append(comodidade)
    precos.append(sum(precos_raw) / quantidade)

# Normalizar os preços usando a normalização Min-Max
scaler = MinMaxScaler()
precos_normalizados = scaler.fit_transform(np.array(precos).reshape(-1, 1)).flatten()  # Correção aqui

# Criar um dicionário que mapeia comodidades para preços normalizados
map_preco_normalizado = {}
for i, comodidade in enumerate(comodidades):
    map_preco_normalizado[comodidade] = precos_normalizados[i]

# Exibir o dicionário
print(map_preco_normalizado)

sum_precosn = sum(map_preco_normalizado.values())

df_copy = df.copy()

for index, row in df.iterrows():
    comodidades = row['amenities']
    comodidades_list = comodidades.split(',')

    comodidades_st = set()
    pesos = []

    for comodidade in comodidades_list:
        comodidade = comodidade.strip()
        comodidade = comodidade.replace('}', '').replace('{', '').replace('-', '').replace('/', '')
        comodidade = comodidade.lower()

        if any(c.isalpha() for c in comodidade) and comodidade in map_preco_normalizado:
            comodidades_st.add(comodidade)


    df_copy.at[index, 'rating_amenities'] = sum(map_preco_normalizado.get(comodidade, 0) for comodidade in comodidades_st)




df = df_copy.copy()

#4 - Normalizar algumas colunas

# Encontre o valor mínimo e máximo na coluna 'rating_amenities'
min_rating = df_copy['rating_amenities'].min()
max_rating = df_copy['rating_amenities'].max()

# Aplicar a fórmula de normalização e atualizar a coluna 'rating_amenities'
df_copy['rating_amenities'] = (df_copy['rating_amenities'] - min_rating) / (max_rating - min_rating)


df_copy['rating_amenities'].describe()


# Encontre o valor mínimo e máximo na coluna 'rating_amenities'
min_rating = df_copy['price'].min()
max_rating = df_copy['price'].max()

# Aplicar a fórmula de normalização e atualizar a coluna 'rating_amenities'
df_copy['price_normal'] = (df_copy['price'] - min_rating) / (max_rating - min_rating)


#5 - Arrumar alguns dados

print(df_copy.isna().sum())

pode = 0
if pode:
  #acha a porcentagem que eu quero
  limit = 0.1
  percentage_values = df_copy['price_normal'].value_counts(normalize=True) * 100
  pd.options.display.max_rows = None
  valores_acima_limite = percentage_values[percentage_values.values >= limit]
  valores_abaixo_limite = percentage_values[percentage_values.values < limit]
  print(percentage_values)

df_selecionado = df_copy.copy()
print(len(df_selecionado))

df_selecionado.describe().transpose()


#6 - Implementação

Codigo de maior puntuação (Random Forest Regressor) Todos os Valores de 'Price'

if True:
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor



  def avaliar_desempenho(df_treinamento):
      df_treinamento = df_treinamento.dropna()

      X = df_treinamento.drop("price", axis=1)
      y = df_treinamento["price"]

      X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

      modelo = RandomForestRegressor()

      modelo.fit(X_treino, y_treino)

      previsoes = modelo.predict(X_teste)

      mse = mean_squared_error(y_teste, previsoes)
      r2 = r2_score(y_teste, previsoes)
      mae = mean_absolute_error(y_teste, previsoes)

      return mse, r2, mae


  colunas_fixas = ["review_scores_location",'latitude','longitude', "rating_amenities", "minimum_nights","price","accommodates","guests_included", 'beds','bedrooms','bathrooms', "maximum_nights",'property_type','room_type','bed_type']


  colunas_para_treinamento = colunas_fixas

  df_treinamento = df_selecionado[colunas_para_treinamento]

  mse, r2, mae = avaliar_desempenho(df_treinamento)

  print("Melhor R^2:", r2)
  print("Melhor MSE:", mse)
  print("Melhor MAE:", mae)

#Codigo de maior puntuação (Random Forest Regressor) Valores de moiores que 0.1 'Price'

pode = 1
if pode:
  #acha a porcentagem que eu quero
  limit = 0.1
  percentage_values = df_copy['price_normal'].value_counts(normalize=True) * 100
  pd.options.display.max_rows = None
  valores_acima_limite = percentage_values[percentage_values.values >= limit]
  valores_abaixo_limite = percentage_values[percentage_values.values < limit]

#essa parte do codigo retira valores extremos
#coloca o cara ja com a porcentagem
df_copy = df_copy[df_copy['price'] < 60000]
df_selecionado = df_copy[df_copy['price_normal'].isin(valores_acima_limite.index)].copy()
df_selec_abaixo = df_copy[df_copy['price_normal'].isin(valores_abaixo_limite.index)].copy()
print(len(df_selecionado))

if True:
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor



  def avaliar_desempenho(df_treinamento):
      df_treinamento = df_treinamento.dropna()

      X = df_treinamento.drop("price", axis=1)
      y = df_treinamento["price"]

      X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

      modelo = RandomForestRegressor()

      modelo.fit(X_treino, y_treino)

      previsoes = modelo.predict(X_teste)

      mse = mean_squared_error(y_teste, previsoes)
      r2 = r2_score(y_teste, previsoes)
      mae = mean_absolute_error(y_teste, previsoes)

      return mse, r2, mae


  colunas_fixas = ["review_scores_location",'latitude','longitude', "rating_amenities", "minimum_nights","price","accommodates","guests_included", 'beds','bedrooms','bathrooms', "maximum_nights",'property_type','room_type','bed_type']


  colunas_para_treinamento = colunas_fixas

  df_treinamento = df_selecionado[colunas_para_treinamento]

  mse, r2, mae = avaliar_desempenho(df_treinamento)

  print("Melhor R^2:", r2)
  print("Melhor MSE:", mse)
  print("Melhor MAE:", mae)
