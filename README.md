# V desafio em Ciências de Dados PUC GO

## Equipe:

<div>
  
  [<img src="https://avatars.githubusercontent.com/u/98399932?v=4" alt="João Victor Porto" width="100">](https://github.com/Joao-vpf)
  [<img src="https://avatars.githubusercontent.com/u/104952737?v=4" alt="João Pedro Lemes" width="100">](https://github.com/Lixomensch)
  [<img src="https://avatars.githubusercontent.com/u/136506636?v=4" alt="João Augusto" width="100">](https://github.com/tenma2010)
  [<img src="https://avatars.githubusercontent.com/u/79798116?v=4" alt="João Marcos" width="100">](https://github.com/JohnMarcosP)
  [<img src = "https://avatars.githubusercontent.com/u/147951553?v=4" alt ="Ludmila" width = "100">](https://github.com/LudLES-t)
  
</div>

## Projeto:

### Sobre o desafio:

O mercado de hospedagem virtual, também conhecido como hosting, é uma indústria em constante crescimento, com uma ampla gama de serviços oferecidos por inúmeras empresas. A competitividade nesse setor é intensa, com empresas frequentemente ajustando seus preços para atrair e reter clientes. Nesse contexto, iremos analisar os dados do Airbnb referentes à cidade do Rio de Janeiro e ver quais insights podem ser extraídos a partir desses dados brutos. A previsão de preços desempenha um papel crucial para as empresas que desejam permanecer competitivas. O problema abordado neste projeto é o desenvolvimento de um modelo de previsão de preços para o mercado de hospedagem virtual.

### Motivo do desafio:

O desafio busca promover o trabalho em equipe, análise exploratória de dados, filtragem de dados, modelagem e algoritmos, aprendizado de linguagens de programação, machine learning e raciocínio lógico, com o objetivo de analisar os dados.

O escopo deste projeto envolve a criação de um modelo de previsão de preços no mercado de hospedagem virtual. Especificamente, o projeto visa:

1. **Limpeza e Pré-processamento de Dados:** Realizar a limpeza e pré-processamento dos dados, incluindo o tratamento de valores ausentes, a codificação de variáveis categóricas e a normalização, quando necessário.

2. **Análise Exploratória de Dados:** Realizar uma análise exploratória para entender a distribuição dos preços, identificar tendências sazonais e correlações entre os diferentes atributos e os preços.

3. **Desenvolvimento do Modelo de Previsão:** Selecionar e desenvolver um modelo de previsão apropriado para estimar os preços no mercado de hospedagem virtual. Isso pode envolver o uso de técnicas de aprendizado de máquina, como regressão, séries temporais ou algoritmos de análise de texto, dependendo da natureza dos dados.

4. **Treinamento e Avaliação do Modelo:** Treinar o modelo com dados históricos e avaliá-lo usando métricas adequadas, como erro médio absoluto (MAE), erro médio quadrático (MSE) ou R².


### Dados:

[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1jtZ8q0LG3WczgN6ORPzajlErQatH7_3p/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Dados%20pre_processados-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1ix98wju56E6pguswDQhCuiLyve-AKCIi/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Roteiro%20do%20projeto%20-%20PDF%20-%20red?style=for-the-badge&logo=files&logoColor=red
)](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Roteiro%20para%20análise%20dos%20dados%20%20desafio%20V%20%20completo.pdf)  

### Codigo:

[![Static Badge](https://img.shields.io/badge/C%C3%B3digo%20do%20projeto-Link-orange?style=for-the-badge&logo=googlecolab)
](https://colab.research.google.com/drive/1uCbaxdK39zXcpc2FMXvMa06_0hzMAiBD?usp=sharing)  [![Static Badge](https://img.shields.io/badge/todos%20os%20algoritmos%20usados-Link-blue?style=for-the-badge)
](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Code/explicacao.md)

## Implementação:

A implementação do desafio foi dividida em várias etapas para lidar com os dados brutos, uma vez que eles contêm muitos ruídos e informações desnecessárias que podem afetar o cálculo e, consequentemente, a qualidade da resposta. As etapas da implementação foram as seguintes:

1. Pre-processamento dos dados;
2. Limpeza dos dados;
3. Completar/corrigir colunas;
4. Normalizar colunas;
5. Finalizar/calcular resposta;

### 1. Pre-processamento dos dados:

1. Foram deletadas todas as colunas que faziam referência ao host do Airbnb, pois o host não possui relação direta com a localização; por exemplo, as colunas de AA a AW foram removidas.
2. Sumários, descrições, IDs e URLs foram eliminados, uma vez que não afetam o cálculo do preço, mas sim a escolha do cliente.
3. Nomes gerais, como cidade, estado e bairro, foram excluídos, uma vez que podem ser interpretados com mais facilidade e precisão a partir das coordenadas de latitude e longitude.
4. Colunas com filtros únicos, como a coluna de confirmação de avaliação e a localização exata, foram removidas, uma vez que não são muito relevantes para as colunas de referência.

#### Extra:

1. Foram corrigidos latitude e longitude para ficarem formatados como numero.

### 2. limpeza dos dados em codigo:

1. Algumas colunas que não foram eliminadas no pré-processamento tiveram que ser deletadas, tais como as colunas "review_scores_cleanliness," "review_scores_checkin," e "review_scores_communication."
2. Foi aplicada uma conversão geral para tornar os valores numéricos.
3. Foram removidos símbolos especiais como "$" e "," das colunas com preços.
4. Limpar colunas de latitude e longitude removendo símbolos especiais como ",", ".", "e+" e "e-", além disso truncar para 10 digitos (quantidade correta para uma boa precisão).

### 3. Completar/corrigir colunas:

#### A) Produzir faltantes na coluna "review_scores_location":

1. Utilizar o modelo [KNeighborsRegressor](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#kneighborsregressor) com o K = 2 e com metrica Euclidiana, com objetivo de relacionar as coordenadas proximas e calcular review_scores_location
   
2. Para o validar o K = 2 foi utilizado o Cross-validation gerando o grafico a seguir:
   
![Melhor K](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/Valor%20de%20K.png)

3. Para o validar a metrica Euclidiana foi utilizado o Cross-validation gerando o grafico a seguir:
   
![Melhor metrica e K](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/Cross-metric-k.png)

4. A review_scores_location que o modelo KNeighborsRegressor gerou o grafico a seguir:
   
![KNN-map](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/KNN-map.png)

5. Gif da KNeighborsRegressor construindo o grafico:
   
![GIF-KNN](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/scatter_animation.gif)

##### Extra:

Para produzir os valores faltantes na coluna "review_scores_location", várias ideias foram consideradas, incluindo o uso de matrizes, vetorização de pontos e algoritmos gananciosos baseados na métrica euclidiana. No entanto, apenas duas ideias se mostraram relevantes: a utilização de uma [KdTree](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#KdTree) e o modelo KNeighborsRegressor. A [implementação da KdTree](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Code/explicacao.md#4--arrumar-review_scores_location-com-kdtree) foi realizada primeiro, mas essa abordagem consumia de 5 a 9 minutos para produzir uma resposta, além de gerar respostas com valores insatisfatórios, com R^2 variando entre 0.7 e 0.8, e o MAE em torno de 200 pontos. Devido a essas métricas, optou-se pela utilização do modelo KNeighborsRegressor, que se mostrou mais eficiente e produziu respostas mais satisfatórias. Grafico gerado usando KdTree:

![Kdtreemap](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/kdtreemap.png)


#### B) Coluna amenities:

A coluna mencionada foi a mais problemática, uma vez que os dados continham muito ruído, informações semelhantes e havia muitos dados em cada item. Portanto, a melhor abordagem para normalizá-la foi salvar todos os tipos diferentes de comodidade e todos os preços das residências em que ela aparecia, com o objetivo de criar um rating que avalie as comodidades de uma residência.  
Para a análise inicial das comodidades, foram produzidos gráficos das comodidades mais comuns nas residências. Um exemplo de visualização é apresentado abaixo, que representa as 20 comodidades mais populares:


![grafico-amenities](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/amenities.png)


Após isso, foram feitas a [normalização](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#normaliza%C3%A7%C3%A3o) de cada comodidade, criando uma relação de comodidade e o valor normalizado, gerando assim o gráfico a seguir:


![grafico-amenities-normalizado](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/amenities_normalizado.png)


E para a conclusão, foi necessário passar novamente em cada item e calcular o somatório dos preços normalizados, gerando assim a coluna "rating_amenities", fazendo referência à qualidade das comodidades, que gerou o gráfico a seguir:


![grafico-amenities-rating](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/ratingxprice.png)


##### Observação sobre o gráfico acima:
É possivel notar que os maiores preços não necessáriamente são os que tem maiores ratings já que dependem de outros fatores para ter o maior preço.


### 4. [Normalizar](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#normaliza%C3%A7%C3%A3o) colunas:

Os dados foram normalizados usando a técnica "MIN-MAX" em duas colunas principais: "rating_amenities" e "price". A coluna "rating_amenities" foi normalizada para facilitar a identificação da qualidade daquela informação e sua importância nos cálculos no [modelo usado](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#avalia%C3%A7%C3%A3o-da-precis%C3%A3o-e-confian%C3%A7a). A coluna "price" foi normalizada para testes e visualização, com o objetivo de determinar a porcentagem que cada valor refletia no total, criando assim a coluna "price_normal" que foi utilizada para o funcionamento do [modelo com maior precisão](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#segunda-an%C3%A1lise-utiliza%C3%A7%C3%A3o-de-valores-da-coluna-price-sem-valores-extremos). Como não houve diferenças entre a quantidade de elementos e sim na escala aqui esta um grafico das comodidades mostrando a diferença entre as escalas da coluna "rating_amenities" como exemplo:

![diferencaamenities](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/diferen%C3%A7asamenities.png) 

### 5. Finalizar/calcular resposta:

#### Revisão de Dados e Análises Anteriores:  

Após a análise exploratória e a normalização das colunas, chegamos ao consenso de que as melhores colunas a serem consideradas são:
- "review_scores_location"
- "latitude"
- "longitude"
- "rating_amenities"
- "minimum_nights"
- "price"
- "accommodates"
- "guests_included"
- "beds"
- "bedrooms"
- "bathrooms"
- "maximum_nights"
- "property_type"
- "room_type"
- "bed_type"
  
Essa seleção foi baseada na ideia de que, com essas variáveis, conseguiríamos explorar todos os aspectos relevantes de uma residência em busca de nos aproximarmos do valor ideal da predição de preços. Essas colunas abrangem tanto características das acomodações quanto informações geográficas, o que é fundamental para a qualidade do modelo de previsão.

#### Seleção de Abordagem e Método:

Em nossa pesquisa em busca de métodos para a solução do projeto, passamos por diversos tipos, como Regressão Linear (clássica), Regressão de Árvore de Decisão, Gradient Boosting, entre outros. Após vários testes e uma pesquisa intensa, optamos por utilizar o **Random Forest Regressor**.

Essa escolha foi baseada na boa consistência demonstrada em diferentes testes e, principalmente, na capacidade de alcançar resultados bastante precisos na predição dos preços. O algoritmo **Random Forest Regressor** é uma escolha sólida para esse tipo de tarefa de previsão de preços devido à sua capacidade de lidar com dados complexos e variáveis independentes.

#### Avaliação da Precisão e Confiança:

Após a aplicação do método **Random Forest Regressor**, observamos resultados bastante satisfatórios que nos fornecem insights valiosos sobre a capacidade do modelo de previsão de preços. Vamos analisar e interpretar os resultados obtidos em duas abordagens diferentes.

**Primeira Análise: Utilização de Todos os Valores da Coluna "price"**

Na primeira análise, utilizando todos os valores da coluna "price," o modelo apresentou os seguintes resultados:

- R²: 0.93
- MSE (Erro Quadrático Médio): 220543.90
- MAE (Erro Absoluto Médio): 50.45

Esses resultados indicam que o modelo teve um desempenho notável. O R² de 0.93 sugere que o modelo é capaz de explicar aproximadamente 93% da variabilidade dos preços, o que é uma precisão significativa. Além disso o MAE de 50.45 são indicativos de que o modelo, em média, erra os preços em cerca de 50R$.

#### Segunda Análise: Utilização de Valores da Coluna "price" Sem Valores Extremos

Na segunda análise, retirando valores extremos da coluna "price," o modelo apresentou os seguintes resultados:

- R²: 0.97
- MSE (Erro Quadrático Médio): 3936.00
- MAE (Erro Absoluto Médio): 15.93

Os resultados desta segunda análise são ainda mais impressionantes. O R² de 0.97 reflete uma precisão notável, indicando que o modelo é capaz de explicar aproximadamente 97% da variabilidade dos preços. Além disso, o MAE de 15.93 demonstram que, em média, o modelo erra os preços em apenas 16R$. 

Esses valores indicam que o modelo é altamente capaz de prever os preços com grande precisão, o que é significativo e valioso para o problema em questão.

#### Conclusão e Implicações Práticas:

Com base nas análises, podemos concluir que o modelo **Random Forest Regressor** é altamente eficaz na previsão de preços no mercado de hospedagem virtual. Seus resultados sugerem que é capaz de fornecer previsões precisas, com uma capacidade notável de explicar a variação dos preços.

Essa precisão é crucial para o mercado de hospedagem virtual, onde pequenas diferenças de preço podem influenciar significativamente as decisões dos clientes. Com um erro absoluto médio de apenas 16R$, o modelo é uma ferramenta valiosa para empresas de hospedagem virtual, permitindo ajustar estratégias de preços com base em previsões altamente confiáveis.

Esses resultados também destacam a importância de pré-processar os dados e remover valores extremos, o que pode melhorar significativamente a precisão do modelo. No entanto, é importante manter a vigilância contínua do desempenho do modelo e realizar atualizações conforme necessário para garantir que ele continue sendo uma ferramenta eficaz no mercado em constante mudança.

## Explicação dos termos usados:

### KNeighborsRegressor:

O KNeighborsRegressor é um algoritmo de regressão que faz parte do conjunto de algoritmos de aprendizado de máquina conhecidos como "k-Nearest Neighbors" (KNN) ou "k-Vizinhos Mais Próximos". Ele é usado para realizar tarefas de regressão, onde o objetivo é prever um valor numérico com base em um conjunto de atributos ou características, a ideia básica por trás do KNN é bastante simples, para fazer uma previsão em novo ponto de dados o algoritmo procura os k pontos de dados mais próximos no conjunto de treinamento, com base em alguma medida de distância, como a distância Euclidiana, em seguida, ele calcula a média (ou em alguns casos a mediana) dos valores-alvo desses k pontos mais próximos para fazer a previsão para o novo ponto de dados.

### KdTree: 

Uma árvore k-d (k-dimensional) é uma estrutura de dados usada para particionar espaços multidimensionais em subespaços menores para facilitar a busca eficiente, essa estrutura é especialmente útil em problemas onde você precisa encontrar os vizinhos mais próximos de um ponto em um espaço multidimensional, como é o caso do algoritmo k-Nearest Neighbors (KNN). A árvore k-d é usada para organizar os dados de treinamento de maneira que as consultas para encontrar os vizinhos mais próximos sejam eficientes, o seu funcionamento comença particionando repetidamente o espaço multidimensional ao longo das dimensões alternadas, cada nó da árvore representa um hiperplano que divide o espaço em dois subespaços menores e os pontos de dados são armazenados nas folhas da árvore.

Para realizar uma consulta de "vizinhos mais próximos" usando uma árvore k-d, você percorre a árvore da seguinte maneira:

1. Comece na raiz da árvore.
2. Compare o ponto de consulta com o hiperplano representado pelo nó atual.
3. Escolha o subespaço que contém o ponto de consulta com base na comparação.
4. Recursivamente, continue a busca no subnó que contém o ponto de consulta, até atingir uma folha da árvore.
5. Na folha, você tem um conjunto de pontos de dados que estão próximos ao ponto de consulta. Você pode calcular as distâncias e encontrar os vizinhos mais próximos.

### RandomForestRegressor:

O RandomForestRegressor é um algoritmo de regressão que faz parte da família de métodos de ensemble. Ele se baseia na construção de múltiplas árvores de decisão para criar previsões mais precisas e robustas. Este algoritmo é uma extensão do RandomForestClassifier, que é amplamente utilizado em problemas de classificação.

Aqui estão algumas das principais características e princípios do RandomForestRegressor:
1. Ensemble de Árvores de Decisão: O algoritmo cria um conjunto (ensemble) de árvores de decisão. Cada árvore é treinada com uma amostra aleatória dos dados e/ou um subconjunto aleatório de recursos. Essa aleatoriedade na seleção de dados e características ajuda a evitar o overfitting, tornando o modelo mais geral.
2. Agregação de Previsões: Para fazer uma previsão, o RandomForestRegressor agrega as previsões de todas as árvores no conjunto. Normalmente, a média das previsões é usada como a previsão final. Essa abordagem de agregação ajuda a reduzir a variância e a melhorar a estabilidade do modelo, tornando as previsões mais robustas.
3. Redução de Overfitting: A aleatoriedade introduzida na construção das árvores e na seleção de dados e recursos ajuda a reduzir o overfitting. Cada árvore individual pode estar sujeita a overfitting, mas, quando combinadas, as previsões tendem a ser mais confiáveis e menos suscetíveis a ruído nos dados.
4. Flexibilidade e Poder Preditivo: O RandomForestRegressor é flexível e pode lidar com uma variedade de tipos de dados e problemas de regressão. Ele é particularmente eficaz quando os dados contêm relações complexas e não lineares.

### Normalização:

A técnica usada para normalização no codigo foi a "Normalização Min-Max", a normalização Min-Max é uma técnica comum em pré-processamento de dados, frequentemente usada para escalar os valores de recursos para um intervalo específico, no caso do do projeto foi de 0 a 1. A ideia algoritmica é a seguinte:

1. Encontre o valor mínimo (min) e máximo (max) dos dados originais.
2. Aplique a seguinte fórmula para normalizar cada valor no intervalo [0, 1]:
```
valor_normalizado = (valor_original - min) / (max - min)
```
3. Os valores normalizados agora estão no intervalo [0, 1], onde o valor mínimo original é mapeado para 0, o valor máximo original é mapeado para 1, e os valores intermediários são escalados de acordo.

A normalização Min-Max é útil quando você deseja garantir que os valores de diferentes recursos estejam em uma escala semelhante, o que pode ser importante em algoritmos de aprendizado de máquina que dependem das magnitudes dos recurso, ela também é usada frequentemente em visualizações e análises para facilitar a comparação dos dados.
