# Antes de comecar,
# Inserir aqui a pasta local do repositorio:

#cd d:/repositorios/Deteccao_de_gastos_municipais_em_volume_muito_superior_ao_padrao_esperado/
import os
os.chdir('d:/repositorios/Deteccao_de_gastos_municipais_em_volume_muito_superior_ao_padrao_esperado/')

# Importa funcoes:
from src.funcoes import *

%matplotlib

# Define pasta de origem dos dados:
pasta = './data_full/'
arquivo = 'PIB_RS_2015-alterado-2.xlsx'

#agrupamento dos municipios parecidos:
# arquivo = 'X:/DCF/SAICE/SAE-I/pub/Piloto análise de dados/PIB_RS_2015-alterado.xlsx'
municipios = pd.read_excel(pasta+arquivo)
municipios = municipios.set_index('nm_munic')

# municipios = pd.get_dummies(municipios) #one hot encoding
municipios = municipios.select_dtypes(['number']) #somente colunas numericas
del municipios['PIB_per_capita'] #retira PIB_per_capita, que eh redundante

scaler = StandardScaler()
municipios_std = scaler.fit_transform(municipios)
municipios_std = DataFrame(municipios_std,index=municipios.index,columns=municipios.columns)

# the ELBOW method:
results = {}
for k in range(1, 15):
    agrupamentos = KMeans(n_clusters=k,random_state=999,n_jobs=-1)
    results[k] = agrupamentos.fit(municipios_std).inertia_
    print('k:',str(k),'inertia:',results[k])

plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.show()
