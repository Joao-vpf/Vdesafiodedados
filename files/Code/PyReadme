1- Leitura de dados:
  -dados = pd.read_excel('drive/MyDrive/Colab Notebooks/total_mod.xlsx'): 
      Lê um arquivo Excel chamado 'total_mod.xlsx' e armazena os em um objeto 'dados'

2-Processamento de Dados:
  -Criação de um objeto chamado 'dados_limpos' a partir de 'dados
  -dados_limpos = dados.copy(): Cria uma cópia do objeto 'dados'
  -dados_limpos = dados_limpos.dropna(how='all'): 
      Remove linhas completamente vazias do objeto 'dados_limpos'

  -dados_limpos.drop(["review_scores_cleanliness", "review_scores_checkin", "review_scores_communication"], axis=1, inplace=True): 
    Remove as colunas especificadas do objeto 'dados_limpos'

  -dados_limpos = dados_limpos.apply(lambda x: pd.to_numeric(x, errors='ignore')):
      Converte todos os valores possíveis em valores numéricos no objeto 'dados_limpos'

  -Corrige colunas com símbolos '$' e ','
  -Mapeia colunas de tipos de propriedades para valores numéricos

3-Imputação de Dados Ausentes:
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

4- Análise e Visualização de Dados:
(todo o codigo a baixo está em FALSE)
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
  
    X_train, X_test, y_train, y_test = train_test_split(df_copy[['latitude', 'longitude']], df_copy['review_scores_location'], test_size=0.2, random_state=42)
  
    imputed_test_scores = knn_model.predict(X_test)
  
    mse = mean_squared_error(y_test, imputed_test_scores)
  
    r2 = r2_score(y_test, imputed_test_scores)
  
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')
  
    -O conjunto de dados é dividido em conjuntos de treinamento e teste usando a função train_test_split. O KNN foi treinado com base nos dados do conjunto de treinamento
    -A métrica de erro médio quadrático (MSE) é calculada para avaliar a diferença entre as previsões do modelo e os valores reais
    -Os valores de MSE e R² são impressos na saída para avaliar a precisão do modelo KNN

5- Criação da lista comodidades:

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

6-Manipulação da lista comodidades:

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

6-Manipulação da lista comodidades:

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
  
7-Dummy

  if False:
    colunas_dummy = ["property_type","room_type", "bed_type"]
    df_copy = pd.get_dummies(df_copy, columns=colunas_dummy)
    df_copy
/*
  -colunas_dummy = ["property_type", "room_type", "bed_type"]: armazena as colunas que o objeto deseja processar
  -df_copy = pd.get_dummies(df_copy, columns=colunas_dummy): o 'pd.get_dummies' cria variáveis dummy para as colunas pré estabelecidas em 'colunas_dummy' onde cada categoria única nas colunas especificadas é convertida em colunas binárias (0 ou 1) com base na presença ou ausência dessa categoria em cada linha.
*/

  if False:
    colunas_tipo = list(df_copy.loc[:, 'property_type_0':'bed_type_4'].columns)
    print(colunas_tipo)
  print(df_copy.isna().sum())

/*
  -colunas_tipo = list(df_copy.loc[:, 'property_type_0':'bed_type_4'].columns): a nova lista 'colunas_tipo' é preenchida com nomes das colunas do objeto 'df_copy' que estão no intervalo das colunas 'property_type_0' a 'bed_type_4'.
  -list(): converte o nome das colunas em uma lista de strings
*/

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

8-IA

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

