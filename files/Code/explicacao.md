# Nesse README ficara uma explicação breve de todo o codigo:

## Codigo usado:

### 1- Leitura de dados: 

Lê um arquivo Excel chamado 'total_mod.xlsx' e armazena os em um objeto 'dados'
```
    dados = pd.read_excel('drive/MyDrive/Colab Notebooks/total_mod.xlsx'): 
```

      
### 2- Processamento de Dados:
```
  df = dados.copy() #dados finais apos o processo de limpeza
  df = df.dropna(how='all')# Remove linhas completamente vazias
  df.drop(["review_scores_cleanliness", 'review_scores_value','review_scores_rating', 'review_scores_accuracy', "review_scores_checkin", "review_scores_communication","availability_30","availability_60","availability_90","availability_365","extra_people","security_deposit",'weekly_price', 'monthly_price','cleaning_fee','square_feet'], axis=1, inplace=True)#faça o drop dessas colunas
  print(df.isna().sum())
```

### 3- Imputação de Dados Ausentes:
```
  df_copy = df.copy()
  
  df_missing = df_copy[df_copy['review_scores_location'].isna()]
  df_not_missing = df_copy[~df_copy['review_scores_location'].isna()]
  
  knn_model = KNeighborsRegressor(n_neighbors=2)  # 2 vizinhos é o melhor pela validação cruzada
  
  knn_model.fit(df_not_missing[['latitude', 'longitude']], df_not_missing['review_scores_location'])
  
  imputed_scores = knn_model.predict(df_missing[['latitude', 'longitude']])
  
  df_copy.loc[df_copy['review_scores_location'].isna(), 'review_scores_location'] = imputed_scores


  -df_copy = df.copy(): 
      É feita uma cópia do objeto 'df' para 'df_copy', preservando o original
  -df_missing = df_copy[df_copy['review_scores_location'].isna()]: 
      Cria o objeto 'df_missing', contendo apenas as linhas onde a coluna 'review_scores_location' tem valores ausentes (NaN)

  -df_not_missing = df_copy[~df_copy['review_scores_location'].isna()]: 
     Cria o objeto 'df_not_missing', contendo as linhas onde a coluna 'review_scores_location' possui valores inseridos

  -knn_model = KNeighborsRegressor(n_neighbors=2): 
      Cria um modelo de regressão KNN (K-Nearest Neighbors) com um parâmetro n_neighbors definido como 2 (utiliza os 2 vizinhos mais próximos para realizar a previsão)

  -knn_model.fit(df_not_missing[['latitude', 'longitude']], df_not_missing['review_scores_location']): 
      O modelo KNN é treinado usando as coordenadas latitude e longitude das entradas no obejeto 'df_not_missing' como recursos e as avaliações da localização ('review_scores_location') como previsão de valor

  -imputed_scores = knn_model.predict(df_missing[['latitude', 'longitude']): 
      KNN faz previsões nas linhas com valores ausentes no obejto 'df_missing', com base nas coordenadas. Preenchendo assim os valores ausentes na coluna 'review_scores_location' com estimativas calculadas

  -df_copy.loc[df_copy['review_scores_location'].isna(), 'review_scores_location'] = imputed_scores: 
      Os valores previstos são inseridos no objeto 'df_copy' preenchendo as linhas onde 'review_scores_location' estava ausente
```
### 4- Análise e Visualização de Dados:

```
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
  
    X_train, X_test, y_train, y_test = train_test_split(df_copy[['latitude', 'longitude']], df_copy['review_scores_location'], test_size=0.2, random_state=42)
  
    imputed_test_scores = knn_model.predict(X_test)
  
    mse = mean_squared_error(y_test, imputed_test_scores)
  
    r2 = r2_score(y_test, imputed_test_scores)
  
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')
  
    -O conjunto de dados é dividido em conjuntos de treinamento e teste usando a função train_test_split. O KNN foi treinado com base nos dados do conjunto de treinamento
    -A métrica de erro médio quadrático (MSE) é calculada para avaliar a diferença entre as previsões do modelo e os valores reais
    -Os valores de MSE e R² são impressos na saída para avaliar a precisão do modelo KNN
```
### 5- Criação da lista comodidades:

```
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

  -'comodidades_dict' é criado para mapear os itens e informações de preços e contagens
  -As comodidades da propriedade são extraídas da coluna 'amenities' e armazenadas na variável 'comodidades'
  -O preço é extraído da coluna 'price' e armazenado na variável 'price'.Colocando as comodidades e o preço como valores individuais
  -split(','): separa as comodidades com base na virgula
  -strip(): Remove espaços em branco em excesso no início e no final do texto da comodidade.
  -replace('}', ''): Remove qualquer caractere '}' da comodidade.
  -replace('{', ''): Remove qualquer caractere '{' da comodidade.
  -replace('-', ''): Remove quaisquer traços '-' da comodidade.
  -replace('/', ''): Remove quaisquer barras '/' da comodidade.
  -lower(): Converte o texto da comodidade para letras minúsculas
  -if comodidade in comodidades_dict:
    verifica se a comodidade já existe. Caso ja exista o código atualiza as informações correspondentes, caso não exista é criado uma nova comodidade
  -map_preco_normalizado = {}: Após a iteração por todas as comodidades das propriedades, o código cria um mapa chamado 'map_preco_normalizado'
  -print(len(comodidades_dict)): Isso exibe o número de comodidades únicas registradas. Cada comodidade única tem uma entrada
```
### 6- Manipulação da lista comodidades:
```
  min_preco = min(preco for precos, count in comodidades_dict.values() for preco in precos)
max_preco = max(preco for precos, count in comodidades_dict.values() for preco in precos)

for comodidade, (precos, count) in comodidades_dict.items():
    preco_normalizado = (sum(precos) / count - min_preco) / (max_preco - min_preco)
    map_preco_normalizado[comodidade] = preco_normalizado
    print(comodidade, preco_normalizado)
/*
  -Calculo do valor mínimo (min_preco) e o valor máximo (max_preco) dos preços registrados no 'comodidades_dict'
  -for comodidade, (precos, count) in comodidades_dict.items(): Para cada comodidade, ele recupera os preços (precos) e a contagem (count) das comodidade
  -preco_normalizado = (sum(precos) / count - min_preco) / (max_preco - min_preco): 
    A soma dos preços dividida pelo número de preços registrados
    Em seguida, ele subtrai o (min_preco) para garantir que todos os preços estejam na mesma escala.
    Divide pelo intervalo dos preços para normalizar os valores
    Armazena os resultados no objeto 'map_preco_normalizado'
*/

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

  -df.copy() é usada para armazenar as classificações totais das comodidades
  -for index, row in df.iterrows(): itera as linhas do df_copy() para tratamento indivudial
  -comodidades = row['amenities']: Extraindo da coluna amenities
  - total_rating = 0: inicializado como 0. A variável irá calcular o rating das comodidades
  -strip(): Remove espaços em branco no início e no final do texto
  -replace('}', ''): Remove qualquer caractere '}'
  -replace('{', ''): Remove qualquer caractere '{'
  -replace('-', ''): Remove quaisquer traços '-'
  -replace('/', ''): Remove quaisquer barras '/'
  -lower(): Converte o texto para minúsculo
  -if any(c.isalpha() for c in comodidade) and comodidade in map_preco_normalizado: Verifica se a comodidade já existe em map_preco_normalizado
  -total_rating += map_preco_normalizado[comodidade]: é feito o calculo da classificalção da comodidade
  -df_copy.at[index, 'rating_amenities'] = total_rating: armazena as classificações em df_copy() onde o rating é atribuido a coluna 'rating_amenities'
```
### 7- Manipulação da lista comodidades:
```
min_preco = min(preco for precos, count in comodidades_dict.values() for preco in precos)
max_preco = max(preco for precos, count in comodidades_dict.values() for preco in precos)

for comodidade, (precos, count) in comodidades_dict.items():
    preco_normalizado = (sum(precos) / count - min_preco) / (max_preco - min_preco)
    map_preco_normalizado[comodidade] = preco_normalizado
    print(comodidade, preco_normalizado)
/*
  -Calculo do valor mínimo (min_preco) e o valor máximo (max_preco) dos preços registrados no 'comodidades_dict'
  -for comodidade, (precos, count) in comodidades_dict.items(): Para cada comodidade, ele recupera os preços (precos) e a contagem (count) das comodidade
  -preco_normalizado = (sum(precos) / count - min_preco) / (max_preco - min_preco): 
    A soma dos preços dividida pelo número de preços registrados
    Em seguida, ele subtrai o (min_preco) para garantir que todos os preços estejam na mesma escala.
    Divide pelo intervalo dos preços para normalizar os valores
    Armazena os resultados no objeto 'map_preco_normalizado'
*/

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

  -df.copy() é usada para armazenar as classificações totais das comodidades
  -for index, row in df.iterrows(): itera as linhas do df_copy() para tratamento indivudial
  -comodidades = row['amenities']: Extraindo da coluna amenities
  - total_rating = 0: inicializado como 0. A variável irá calcular o rating das comodidades
  -strip(): Remove espaços em branco no início e no final do texto
  -replace('}', ''): Remove qualquer caractere '}'
  -replace('{', ''): Remove qualquer caractere '{'
  -replace('-', ''): Remove quaisquer traços '-'
  -replace('/', ''): Remove quaisquer barras '/'
  -lower(): Converte o texto para minúsculo
  -if any(c.isalpha() for c in comodidade) and comodidade in map_preco_normalizado: Verifica se a comodidade já existe em map_preco_normalizado
  -total_rating += map_preco_normalizado[comodidade]: é feito o calculo da classificalção da comodidade
  -df_copy.at[index, 'rating_amenities'] = total_rating: armazena as classificações em df_copy() onde o rating é atribuido a coluna 'rating_amenities'
 ```

### 8- Definir qual ponto dos dados vai ser usado para treinamento
```
  pode = 0
if pode:
  limit = 0.1
  percentage_values = df_copy['price_normal'].value_counts(normalize=True) * 100
  pd.options.display.max_rows = None
  valores_acima_limite = percentage_values[percentage_values.values >= limit]
  valores_abaixo_limite = percentage_values[percentage_values.values < limit]
  print(percentage_values)
/*
  -percentage_values = df_copy['price_normal'].value_counts(normalize=True) * 100:  
    .calcula a porcentagem de valores na coluna 'price_normal'
    .df_copy. value_counts(normalize=True): conta a frequência de cada valor e multiplica por 100 para obter as porcentagens(essas porcentagens representam a distribuição dos valores nessa coluna)
  -pd.options.display.max_rows = None: configura a biblioteca Pandas para exibir todas as linhas do objeto ao imprimir
  -valores_acima_limite e valores_abaixo_limite: 
    .valores_acima_limite possua as porcentagens que estão acima do valor de 'limit'
    .valores_abaixo_limite possua as porcentagens que estão a baixo do valor de 'limit'
*/

  if pode:
  df_selecionado = df_copy[df_copy['price_normal'].isin(valores_acima_limite.index)].copy()
  df_selec_abaixo = df_copy[df_copy['price_normal'].isin(valores_abaixo_limite.index)].copy()
  print(len(df_selecionado))
  print(len(df_selec_abaixo))
else:
  df_selecionado = df_copy.copy()

/*
  -df_selecionado = df_copy[df_copy['price_normal'].isin(valores_acima_limite.index)].copy(): cria o objeto 'df_selecionado' que recebe os valores do objeto 'valores_acima_limite'
  - df_selec_abaixo = df_copy[df_copy['price_normal'].isin(valores_abaixo_limite.index)].copy(): cria o objeto 'df_selec_abaixo' que recebe os valores do objeto 'valores_abaixo_limite'
*/
```
### 9- Modelo RandomForestRegressor para calcular a resposta
```
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
  df_treinamento = df_selecionado_01[colunas_para_treinamento]
  mse, r2, mae = avaliar_desempenho(df_treinamento)

  print("Melhor R^2:", r2)
  print("Melhor MSE:", mse)
  print("Melhor MAE:", mae)

    -avaliar_desempenho(df_treinamento):
      .Remove todas as linhas que contenham valores ausentes (NaN).
      .X = df_treinamento.drop("price", axis=1): divide em variáveis X
      .y = df_treinamento["price"]: divide em variáveis de destino (Y)
      .train_test_split: divide os conjuntos de treinamento e os conjuntos de teste
      .modelo = RandomForestRegressor(): cria o modelo de regressão floresta aleatória, o modelo é treinado no conjunto de treinamento e cria previsões no conjunto teste
      .'mse = mean_squared_error(y_teste, previsoes)', 'r2 = r2_score(y_teste, previsoes)', 'mae = mean_absolute_error(y_teste, previsoes)': calculam métricas de avaliação
```

## EXTRAS:


### 1- Validação cruzada para achar o melhor K para o KNeighborsRegressor

```
#Cross Validation para achar o melhor K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


# Separar os dados em faltando e não faltando review_scores_location
df_missing = df_copy[df_copy['review_scores_location'].isna()]
df_not_missing = df_copy[~df_copy['review_scores_location'].isna()]

# Definir o intervalo de valores de k a serem testados
k_range = range(1, 11)

# Função para encontrar o melhor valor de k
def find_best_k(df, k_range):
    best_k = None
    best_score = -np.inf

    for k in k_range:
        knn_model = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn_model, df[['latitude', 'longitude']], df['review_scores_location'], cv=5)
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k

# Encontre o melhor valor de k usando os dados não faltantes
best_k = find_best_k(df_not_missing, k_range)
print(f"Melhor valor de k: {best_k}")

# Crie um gráfico para visualizar a validação cruzada
mean_scores = []

for k in k_range:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn_model, df_not_missing[['latitude', 'longitude']], df_not_missing['review_scores_location'], cv=5)
    mean_scores.append(np.mean(scores))

plt.figure(figsize=(10, 6))
plt.plot(k_range, mean_scores, marker='o', linestyle='-')
plt.title("Validação Cruzada para Encontrar o Melhor Valor de K")
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Pontuação Média da Validação Cruzada")
plt.xticks(k_range)
plt.grid(True)
plt.show()

```

### 2- Validação cruzada para metrica e para o K

```
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Valores de k a serem testados
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5],
    'p': [1, 2]  # 1 para Manhattan, 2 para Euclidiana
}

# Criar o modelo k-NN
knn_model = KNeighborsRegressor()

# Criar um objeto GridSearchCV para realizar a validação cruzada
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Separar as features e o target
X = df_copy[['latitude', 'longitude']]
y = df_copy['review_scores_location']

# Realizar a validação cruzada
grid_search.fit(X, y)

# Obter os resultados da validação cruzada
results = grid_search.cv_results_

# Extrair os scores negativos médios
mean_scores = -results['mean_test_score']

# Plotar os resultados
plt.figure(figsize=(10, 6))
for p in param_grid['p']:
    if(p == 1):
      res = 'Euclidiana'
    else:
      res = 'Manhattan'
    plt.plot(param_grid['n_neighbors'], mean_scores[p - 1::2], label=f'Métrica {res}')

plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Erro Quadrático Médio Negativo')
plt.legend()
plt.title('Validação Cruzada k-NN')
plt.show()

# Encontrar os melhores hiperparâmetros
best_params = grid_search.best_params_
best_k = best_params['n_neighbors']
best_metric = 'Manhattan' if best_params['p'] == 1 else 'Euclidiana'
best_score = -grid_search.best_score_

print(f"Melhores hiperparâmetros: k = {best_k}, Métrica = {best_metric}")
print(f"Erro Quadrático Médio Negativo Médio: {best_score:.2f}")
```

### 3- Criar gif

```
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from PIL import Image

# gif
plot_df = df[['latitude', 'longitude', 'review_scores_location']].copy()

colors = ['#941801', '#FC583A',  '#FF7F50', "#fc8f12", "#ffad50", "#DBCD02", "#f6ff8f", '#dfff8f', '#98FB98', '#008000']
cmap = ListedColormap(colors)
# Divida os dados em lotes de 3000 pontos
batch_size = 10000
num_batches = len(plot_df)
# Calcule o número de repetições com base na divisão em lotes
repetitions = num_batches // batch_size

# Criar uma lista para armazenar os nomes dos arquivos de imagem
image_files = []

for i in range(repetitions):
    fig, ax = plt.subplots(figsize=(10, 6))
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(plot_df))
    batch_df = plot_df.iloc[:end_idx]

    scatter = ax.scatter(batch_df['longitude'], batch_df['latitude'], c=batch_df['review_scores_location'], cmap=cmap, s=4)
    plt.colorbar(scatter, ax=ax, label='review_scores_location')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Pontos de rating')

    # Salvar o gráfico como uma imagem
    image_file = f'batch_{i}.png'
    image_files.append(image_file)
    plt.savefig(image_file, dpi=300)  # Salvar a imagem com alta resolução

    plt.close(fig)  # Fechar a figura após salvar

# Criar o GIF animado a partir das imagens com o número de repetições calculado
images = [Image.open(image_file) for image_file in image_files]
images[0].save('scatter_animation.gif', save_all=True, append_images=images[1:], duration=100, loop=repetitions)

# Remover os arquivos de imagem após criar o GIF
for image_file in image_files:
    os.remove(image_file)
```



### 4- Arrumar review_scores_location com KdTREE
```
df_copy = df.copy()
# trocar valores NaN
def avg_score(row, tree, df_copy, valid_index_means, valid_indices):
    if math.isnan(row['review_scores_location']):
        # Consulta as 10 entradas mais próximas com valores de review_scores_location válidos
        distances, indices = tree.query([row['latitude'], row['longitude']], k=30)

        #busca na hash e coloca apenas os indices reais
        permitidos = [indice for indice in indices if indice in valid_indices]

        # Verifica se já calculamos a média para esses índices
        if tuple(permitidos) in valid_index_means:
            average_score = valid_index_means[tuple(permitidos)]
        else:
        #calcula a media caso nao foi calculada
            average_score = df_copy.loc[permitidos, 'review_scores_location'].mean()
            valid_index_means[tuple(permitidos)] = average_score

        return average_score
    else:
        return row['review_scores_location']

# kdtree
valid_lat_lon = df_copy.loc[df_copy['review_scores_location'].notna(), ['latitude', 'longitude']]
tree = cKDTree(valid_lat_lon.values)

# Índices válidos
valid_indices = df_copy.index
valid_indices = valid_indices.to_series().to_dict() #cria uma hash dos indices validos

# Dicionário para armazenar médias calculadas
valid_index_means = {}

# Aplica a função aos dados
df_copy['review_scores_location'] = df_copy.apply(avg_score, args=(tree, df_copy, valid_index_means, valid_indices), axis=1)
df_copy['review_scores_location'] = df_copy['review_scores_location'].round(2)

del tree  # Exclui a árvore cKDTree

# Limpa o dicionário valid_index_means
valid_index_means.clear()
valid_index_means = None

del valid_indices

df_copy


pontos = df_copy[['latitude', 'longitude']].values
kdtree = cKDTree(pontos)

# Pré-calcule min_score e max_score
max_score = df_copy['review_scores_location'].max()
min_score = df_copy['review_scores_location'].min()

def calculate_rating(row):
    dist, indices = kdtree.query([row['latitude'], row['longitude']], k=5)
    nearest_scores = df_copy['review_scores_location'].iloc[indices].to_numpy()  
    normalized_rating = nearest_scores.mean()
    return (normalized_rating - min_score) / (max_score - min_score) * 10

df_copy['rating'] = df_copy.apply(calculate_rating, axis=1).round(2).clip(lower=0, upper=10)

del kdtree  # Exclui a árvore cKDTree
df_copy[['latitude', 'longitude', 'review_scores_location', 'rating']]

plot_df = df_copy[['latitude', 'longitude', 'rating']]

colors = ['#941801', '#FC583A',  '#FF7F50', "#fc8f12", "#ffad50" ,"#DBCD02","#f6ff8f", '#dfff8f','#98FB98', '#008000']
cmap = ListedColormap(colors)

# Criar o gráfico de dispersão
plt.figure(figsize=(10, 6))
scatter = plt.scatter(plot_df['longitude'], plot_df['latitude'], c=plot_df['rating'], cmap=cmap, s=10)
plt.colorbar(scatter, label='Rating')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Rating das coordenadas')
plt.show()
```

### 5- Gerar mapa
```
if True:

  def color(rating):
      rating = int(rating)
      if rating < 20:
          return "red"
      if rating < 50:
          return "orange"
      if rating < 80:
          return "yellow"
      return "green"

  df['review_scores_location'] = pd.to_numeric(df['review_scores_location'], errors='coerce')
  # Filtra e seleciona as colunas relevantes
  df = df.dropna(subset=["longitude", "latitude", "review_scores_location"])
  df = df[["latitude", "longitude", "review_scores_location"]]

  # Calcula as cores com base na pontuação
  df["color"] = df["review_scores_location"].apply(color)

  # Cria um mapa
  m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

  # Crie um cluster de marcadores
  marker_cluster = MarkerCluster().add_to(m)

  # preencha o cluster
  for index, row in df.iterrows():
      folium.CircleMarker(
          location=[row['latitude'], row['longitude']],
          color=row['color'],
          fill=True,
          radius=5,
          popup=f'Review Score: {int(row["review_scores_location"])}'
      ).add_to(marker_cluster)

  m.save('mapa_rating.html')
print(len(df))
df
```



