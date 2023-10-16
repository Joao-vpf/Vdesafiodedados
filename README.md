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

Nesse projeto iremos analisar os dados dos Airbnb referentes à cidade do Rio de Janeiro e ver quais insights podem ser extraídos a partir desses dados brutos.

### Motivo do desafio:

O desafio busca promover o trabalho em equipe, analise exploratória de dados, filtragem de dados, modelagem e algoritmos, aprendizado de linguagens de programação, machine learning e raciocínio lógico perante problemas pertinentes.

### Dados:

[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1jtZ8q0LG3WczgN6ORPzajlErQatH7_3p/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1ix98wju56E6pguswDQhCuiLyve-AKCIi/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Roteiro%20do%20projeto%20-%20PDF%20-%20red?style=for-the-badge&logo=files&logoColor=red
)](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Roteiro%20para%20análise%20dos%20dados%20%20desafio%20V%20%20completo.pdf)  

### Codigo:

[![Static Badge](https://img.shields.io/badge/C%C3%B3digo%20do%20projeto-Link-orange?style=for-the-badge&logo=googlecolab)
](https://colab.research.google.com/drive/1uCbaxdK39zXcpc2FMXvMa06_0hzMAiBD?usp=sharing)  [![Static Badge](https://img.shields.io/badge/Explica%C3%A7%C3%A3o%20do%20codigo-Link-blue?style=for-the-badge)
](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/PyReadme)

## Implementação:

A implementação do desafio foi dividida em várias etapas para lidar com os dados brutos, uma vez que eles contêm muitos ruídos e informações desnecessárias que podem afetar o cálculo e, consequentemente, a qualidade da resposta. As etapas da implementação foram as seguintes:

1. Pre-processamento dos dados;
2. Limpeza dos dados;
3. Completar/corrigir colunas;
4. Normalizar colunas;
5. Finalizar/calcular resposta

### Pre-processamento dos dados:

1. Foram deletadas todas as colunas que faziam referência ao host do Airbnb, pois o host não possui relação direta com a localização; por exemplo, as colunas de AA a AW foram removidas.
2. Sumários, descrições, IDs e URLs foram eliminados, uma vez que não afetam o cálculo do preço, mas sim a escolha do cliente.
3. Nomes gerais, como cidade, estado e bairro, foram excluídos, uma vez que podem ser interpretados com mais facilidade e precisão a partir das coordenadas de latitude e longitude.
4. Colunas com filtros únicos, como a coluna de confirmação de avaliação e a localização exata, foram removidas, uma vez que não são muito relevantes para as colunas de referência.

#### Extra:

1. Foram corrigidos latitude e longitude para ficarem formatados como numero.

### limpeza dos dados em codigo:

1. Algumas colunas que não foram eliminadas no pré-processamento tiveram que ser deletadas, tais como as colunas "review_scores_cleanliness," "review_scores_checkin," e "review_scores_communication."
2. Foi aplicada uma conversão geral para tornar os valores numéricos.
3. Foram removidos símbolos especiais como "$" e "," das colunas com preços.
4. Limpar colunas de latitude e longitude removendo símbolos especiais como ",", ".", "e+" e "e-", além disso truncar para 10 digitos (quantidade correta para uma boa precisão).

### Produzir faltantes na coluna "review_scores_location":

1. Utilizar o modelo [KNeighborsRegressor](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#kneighborsregressor) com o K = 2 e com metrica Euclidiana, com objetivo de relacionar as coordenadas proximas e calcular review_scores_location
2. Para o validar o K = 2 foi utilizado o Cross-validation gerando o grafico a seguir:

3. Para o validar a metrica Euclidiana foi utilizado o Cross-validation gerando o grafico a seguir:

4. A review_scores_location que o modelo KNeighborsRegressor gerou o grafico a seguir:

#### Extra:

Para produzir os valores faltantes na coluna "review_scores_location", várias ideias foram consideradas, incluindo o uso de matrizes, vetorização de pontos e algoritmos gananciosos baseados na métrica euclidiana. No entanto, apenas duas ideias se mostraram relevantes: a utilização de uma [KdTree](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/README.md#KdTree) e o modelo KNeighborsRegressor. A implementação da KdTree foi realizada primeiro, mas essa abordagem consumia de 5 a 9 minutos para produzir uma resposta, além de gerar respostas com valores insatisfatórios, com R^2 variando entre 0.7 e 0.8, e o MAE em torno de 200 pontos. Devido a essas métricas, optou-se pela utilização do modelo KNeighborsRegressor, que se mostrou mais eficiente e produziu respostas mais satisfatórias. Grafico gerado usando KdTree:

![Map-KdTree](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/graficos/KdTreeMap.jpg)

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

