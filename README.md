## Detecção de gastos municipais em volume muito superior ao padrão esperado

O objetivo do trabalho é detectar gastos dos municípios em volume muito acima de um padrão estabelecido para municípios semelhantes, via agregação mensal das liquidações por tipo de gasto, sendo um dos possíveis fatores a serem levados em consideração quando da auditoria, posterior ou concomitante, a fim de identificar irregularidades.

Após agrupar municípios com características semelhantes em clusters, foi efetuada uma modelagem simples das séries temporais históricas para cada tipo de despesa, utilizando tendência e sazonalidade, obtendo-se o padrão esperado para aquele grupo.

Trazendo os gastos de cada município e do padrão esperado para o seu respectivo grupo para uma mesma escala, foi possível identificar anomalias nos gastos municipais, apontando quais conjuntos de liquidações municipais eventualmente distoam do padrão estabelecido para município semelhantes.

## Base de dados

A base de dados deve estar na pasta /data/.

### Utilização

Pode ser rodado tanto no shell do Ipython quanto no Jupyter Notebook.

Antes de começar troque a pasta de trabalho para a sua instalação local do repositório.

Depois, é só executar os comandos do arquivo principal.py (pasta /src/). São importadas as funções do arquivo funções.py (pasta /src/) e depois feita a análise.

Gráficos podem ser gerados para melhor visualização das anomalias através da função plotagem().

### Saída

O sistema gera como saída os seguintes arquivos:

anomalias.csv --> todas as rubricas municipais anômalas (gastos muito acima do padrão esperado para um grupo de municípios semelhante);

stats.csv --> estatísticas diversas;

lançamentos --> pasta com os lançamentos/liquidações de cada município que geraram as anomalias.

## Explicação mais detalhada com gráficos

Explicação detalhada pode ser encontrada no arquivo "Explicação - Detecção de liquidações municipais muito acima do esperado.docx"

Maiores informações sobre o modelo e como treiná-lo estão nos comentários de principal.py

Em caso de dúvidas, falar com o autor, Alexandre Gandini.

## Trabalho desenvolvido sobre a seguinte licença:

CREATIVE COMMONS Attribution-NonCommercial / CC BY-NC
-> Citar a autoria do projeto;
-> Proibido utilizar para fins comerciais.