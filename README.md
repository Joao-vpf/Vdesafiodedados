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

Nesse projeto iremos analisar os dados dos Airbnb referentes à cidade do Rio de Janeiro e ver quais insights podem ser extraídos a partir desses dados brutos.

### Dados:

[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1jtZ8q0LG3WczgN6ORPzajlErQatH7_3p/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://docs.google.com/spreadsheets/d/1ix98wju56E6pguswDQhCuiLyve-AKCIi/edit?usp=sharing&ouid=112578483692686555513&rtpof=true&sd=true)
[![Static Badge](https://img.shields.io/badge/Roteiro%20do%20projeto%20-%20PDF%20-%20red?style=for-the-badge&logo=files&logoColor=red
)](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Roteiro%20para%20análise%20dos%20dados%20%20desafio%20V%20%20completo.pdf)  

### Codigo:

[![Static Badge](https://img.shields.io/badge/C%C3%B3digo%20do%20projeto-Link-orange?style=for-the-badge&logo=googlecolab)
](https://colab.research.google.com/drive/1uCbaxdK39zXcpc2FMXvMa06_0hzMAiBD?usp=sharing)  [![Static Badge](https://img.shields.io/badge/Explica%C3%A7%C3%A3o%20do%20codigo-Link-blue?style=for-the-badge)
](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/PyReadme)

## Explicação da implementação:

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
4. As colunas de latitude e longitude foram limpas, removendo símbolos especiais como ",", ".", "e+" e "e-".

   



