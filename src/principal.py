# Antes de comecar,
# Inserir aqui a pasta local do repositorio:

# PARAMETROS:
qtd_lancamentos_da_rubrica_anormal_a_mostrar = 20
qtd_digitos_rubrica = 8 # Quantidade de digitos a considerar da rubrica
percentual_maximo_de_meses_com_zero_pagamento = 0.30
percentual_maximo_muitos_meses_sem_pagamento_nos_ultimos_meses = 0.50
qtd_meses_atras_pra_retirar_rubricas_com_zero_pagamento = 12
lista_de_rubricas_a_deletar = ['319011'] # Pra tirar a rubrica de folha de pagamento
mes_inicial_da_analise = '201001'
janela_desvio_padrao = 12 #meses
qtd_desvios_padroes = 2 #acima de qts desvios padros indica anomalia
valor_minimo_pra_indicar_anomalia = 50000
mes_a_comecar_alertas = 1
ano_a_comecar_alertas = 2018

#cd d:/repositorios/Deteccao_de_gastos_municipais_em_volume_muito_superior_ao_padrao_esperado/
import os
os.chdir('c:/repositorios/anomalias-gastos-municipais/')

# Importa funcoes:
from src.funcoes import *

import matplotlib.style
# plt.style.use('default')
# plt.style.use('ggplot')
plt.style.use('seaborn')

# Cria diretorios de saida se ainda nao existirem:
if not os.path.isdir('./output/'):
	os.makedirs('./output/')
if not os.path.isdir('./output/PM/'):
	os.makedirs('./output/PM/')
if not os.path.isdir('./output/PM/CMMs_junto_no_arquivo/'):
	os.makedirs('./output/PM/CMMs_junto_no_arquivo/')
if not os.path.isdir('./output/PM/lancamentos/'):
	os.makedirs('./output/PM/lancamentos/')
if not os.path.isdir('./output/CM/'):
	os.makedirs('./output/CM/')
if not os.path.isdir('./temp/'):
	os.makedirs('./temp/')

# Se estiver usando o Ipython/Jupyter notebook:
# %matplotlib

# Define pasta de origem dos dados:
pasta = './data_full/'

# Geracao dos grupos de municipios via K-Means:
municipios_agrupados, grupos = cria_grupos_de_municipios_parecidos_via_k_means(pasta,'PIB_RS_2015-alterado-2.xlsx',qtd_grupos=5,random_seed=999)

# Descobre arquivos de dados do SIAPC:
# pasta = './data/siapc/' #pasta com poucos arquivos, somente para demonstracao
# poder = 'PM' ou 'CM'
# primeiro rodar PM, os lancamentos das CMs que estao dentro do arquivo das PMs sao arquivados na pasta './output/PM/CMMs_junto_no_arquivo/'
# antes de rodar as CMs, copiar estes arquivos .pkl pra pasta data_full/CM
poder = 'PM'
# poder = 'CM'
pasta = pasta + poder + '/'
arquivos = descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.pkl')

# Importa e agrega despesas do SIAPC:
rubricas = gera_series_agrupadas_a_partir_de_arquivos(arquivos,pasta,poder=poder,pasta_a_gravar_CM='./output/PM/CMMs_junto_no_arquivo/',qtd_digitos_rubrica=qtd_digitos_rubrica,coluna_a_manter = 'VL_LIQUIDACAO',agrupador1 = 'DT_LIQUIDACAO',agrupador2 = 'rubrica')
rubricas = arruma_nomes_das_colunas(rubricas)
# Retira rubricas com muitos meses com zero
rubricas = retira_rubricas_com_muitos_meses_sem_pagamento(rubricas=rubricas,percentual_maximo_de_meses_com_zero_pagamento=percentual_maximo_de_meses_com_zero_pagamento)
# Retira rubricas com muitos zeros nos ultimos meses:
rubricas = retira_rubricas_com_muitos_meses_sem_pagamento_nos_ultimos_meses(rubricas=rubricas,qtd_meses_atras=qtd_meses_atras_pra_retirar_rubricas_com_zero_pagamento,percentual_maximo_de_meses_com_zero_pagamento=percentual_maximo_muitos_meses_sem_pagamento_nos_ultimos_meses)
# Pra tirar a rubrica de folha de pagamento:
rubricas = retira_rubricas(rubricas=rubricas,lista_de_rubricas_a_deletar=lista_de_rubricas_a_deletar)
# Gera agrupamento das rubricas dos grupos de municipios:
rubricas_grupos = cria_grupo_agrupado_das_rubricas_comuns_a_todos_os_municipios_parecidos(rubricas,grupos=grupos,retirar_meses_com_zeros=False)

# Limpeza:
mes_inicial = mes_inicial_da_analise
rubricas = define_mes_inicial(rubricas=rubricas, mes_inicial=mes_inicial)
rubricas_grupos = define_mes_inicial(rubricas=rubricas_grupos, mes_inicial=mes_inicial)
ultimo_mes, municipios_com_problemas, mes_da_analise = descobre_erros_no_mes_final(rubricas)

#limita a base ao mes da analise (tem municipios que jah manda antes as proximas remessas):
for municipio in tqdm(rubricas):
	rubricas[municipio] = rubricas[municipio][:mes_da_analise]
for grupo in tqdm(rubricas_grupos):
	rubricas_grupos[grupo] = rubricas_grupos[grupo][:mes_da_analise]
ultimo_mes_grupos = descobre_erros_no_mes_final(rubricas_grupos)
ultimo_mes, municipios_com_problemas, mes_da_analise = descobre_erros_no_mes_final(rubricas)

# Deleta os municipios que tem problemas na remessa - ainda nao enviaram remessa do SIAPC:
for municipio in municipios_com_problemas:
	del rubricas[municipio]

# Standardizacao para comparar series de direntes escalas, nos grupos:
rubricas_std = standardizacao(rubricas)
rubricas_grupos_std = standardizacao(rubricas_grupos)

# Projecao via linear regression trend + sazonalidade:
rubricas_grupos_projetadas, r2_grupos = gera_projecao_via_linear_reg_trend_e_sazonalidade(rubricas_grupos_std)
# Para comparar municipio consigo mesmo:
# rubricas_projetadas, r2 = gera_projecao_via_linear_reg_trend_e_sazonalidade(rubricas_std)

# Define mes de inicio dos alertas:
mes_da_remessa = ultimo_mes.groupby('ultimo_mes').count().sort_values(by='municipio').iloc[-1].name.month
ano_da_remessa = ultimo_mes.groupby('ultimo_mes').count().sort_values(by='municipio').iloc[-1].name.year
qtd_meses_atras = (ano_da_remessa - ano_a_comecar_alertas) * 12 + (mes_da_remessa - mes_a_comecar_alertas)

# Detecta anomalias/outliers:
anomalias_comparando_com_grupo = em_comparacao_com_o_grupo_encontra_rubricas_fora_das_bandas(grupos=grupos,rubricas_std=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,window=janela_desvio_padrao,qtd_std_dev=qtd_desvios_padroes,qtd_meses_atras=qtd_meses_atras+1,rubricas=rubricas)
# Para comparar municipio consigo mesmo:
# anomalias_comparando_com_o_proprio_municipio = em_comparacao_com_o_proprio_municipio_encontra_rubricas_fora_das_bandas(grupos=grupos,rubricas_std=rubricas_std,rubricas_projetadas=rubricas_projetadas,window=12,qtd_std_dev=2,qtd_meses_atras=15)

# Seleciona soh as anomalias acima do projetado (nao as abaixo):
anomalias_comparando_com_grupo = anomalias_comparando_com_grupo[anomalias_comparando_com_grupo['qtd_desvios'] > 0]
# Para comparar municipio consigo mesmo:
# anomalias_comparando_com_o_proprio_municipio = anomalias_comparando_com_o_proprio_municipio[anomalias_comparando_com_o_proprio_municipio['high/low'] == 'high']

#seleciona soh as anomalias acima de um determinado valor
anomalias_comparando_com_grupo = anomalias_comparando_com_grupo[anomalias_comparando_com_grupo['valor_total_rubrica'] >= valor_minimo_pra_indicar_anomalia]
# Para comparar municipio consigo mesmo:
# anomalias_comparando_com_o_proprio_municipio = anomalias_comparando_com_o_proprio_municipio[anomalias_comparando_com_o_proprio_municipio['valor_total_rubrica'] >= 50000]

# Adiciona significado das rubricas:
anomalias_comparando_com_grupo = adiciona_significado_da_rubrica_ate_elemento_nas_anomalias(anomalias_comparando_com_grupo,arquivo='./data/Tabelas Auxiliares.xlsx')
anomalias_comparando_com_grupo = anomalias_comparando_com_grupo.sort_values(['municipio','mes','rubrica'],ascending=[True,False,True])
anomalias_comparando_com_grupo.to_csv('./output/'+poder+'/'+'anomalias'+'.csv',decimal=',',sep=';',encoding='utf-8-sig')

# DEMORA 1h40min:
# Mostra os lancamentos que geraram as rubricas anomalas
lancamentos, erros = mostra_lancamentos_das_rubricas_anormais(anomalias=anomalias_comparando_com_grupo,pasta=pasta,arquivos=arquivos,qtd_lancamentos=qtd_lancamentos_da_rubrica_anormal_a_mostrar,qtd_meses_atras=qtd_meses_atras,mes_da_analise=mes_da_analise,poder=poder,qtd_digitos_rubrica=qtd_digitos_rubrica)
# Para comparar municipio consigo mesmo:
# lancamentos, erros = mostra_lancamentos_das_rubricas_anormais(anomalias=anomalias_comparando_com_o_proprio_municipio,pasta=pasta,arquivos=arquivos,qtd_lancamentos=3,qtd_meses_atras=15,)

pasta_temp = './temp/'
# grava(lancamentos,filename='lancamentos.pkl',path=pasta_temp)
# grava(erros,filename='erros.pkl',path=pasta_temp)

lancamentos = abre(filename='lancamentos.pkl',path=pasta_temp)
erros = abre(filename='erros.pkl',path=pasta_temp)

# Adiciona significado das rubricas:
lancamentos = adiciona_significado_da_rubrica_ate_elemento(lancamentos=lancamentos,arquivo='./data/Tabelas Auxiliares.xlsx')

# Exporta lancamentos que geraram as rubricas anomalas:
gera_arquivo_exportacao_csv(lancamentos=lancamentos,pasta='./output/'+poder+'/lancamentos/')

# Gera arquivo de estatisticas:
stats = gera_estatisticas(anomalias_comparando_com_grupo,grupos)
stats.to_csv('./output/'+poder+'/stats.csv',decimal=',',sep=';',encoding='utf-8-sig')

#PLOTAGEM:

#realizado do municipio + bandas
plotagem(projetado_modelo=False,bandas=True,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#modelo do grupo + bandas.
plotagem(projetado_modelo=True,bandas=True,realizado=False,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#realizado do municipio + projetado do modelo:
plotagem(projetado_modelo=True,bandas=False,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#somente realizado do municipio:
plotagem(projetado_modelo=False,bandas=False,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#somente projetado do modelo:
plotagem(projetado_modelo=True,bandas=False,realizado=False,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#somente bandas:
plotagem(projetado_modelo=False,bandas=True,realizado=False,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
#somente realizado do grupo:
plotagem(projetado_modelo=False,bandas=False,realizado=False,realizado_grupo=True,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)

####

anomalias_comparando_com_grupo.sort_values('qtd_desvios')[['rubrica','municipio','mes','qtd_desvios']].tail(20)

#exemplos selecionados previamente:
exemplos = {'ARATIBA':'335043','NOVO HAMBURGO':'339047','SAPUCAIA DO SUL':'469071','PORTO ALEGRE':'449052','CAXIAS DO SUL':'339039','SAO PEDRO DO BUTIA':'319008','ALEGRETE':'449051','BENTO GONCALVES':'319008','CAXIAS DO SUL':'339093','CAMAQUA':'449051','CANOAS':'339039','CANELA':'339093','OSORIO':'339047','BOSSOROCA':'319008','BARRACAO':'335041','ACEGUA':'339046','CARAZINHO':'335043'}

#exemplos de anomalias no ultimo mes da base (201804):
#entre 0.5 e 2 desvios padrão:
exemplos = {'MONTE ALEGRE DOS CAMPOS':'339039','MARIANO MORO':'319004','GRAVATAI':'339092','SAO JERONIMO':'339047','BARAO':'339030','CARAZINHO':'339036','SAO MARTINHO DA SERRA':'339039','CAPIVARI DO SUL':'335043','LIBERATO SALZANO':'335043','CANOAS':'339032','GRAVATAI':'339032','TAPES':'469071','BOA VISTA DO SUL':'319113','SAO NICOLAU':'339030','NOVO CABRAIS':'339032','GENERAL CAMARA':'339030','HERVAL':'319008','SAO JERONIMO':'339049','BOM PRINCIPIO':'469071','RIOZINHO':'339039','IMIGRANTE':'339039','VERANOPOLIS':'335041','NOVO TIRADENTES':'339048','CAIBATE':'339030','FARROUPILHA':'469071','JULIO DE CASTILHOS':'339032','SANANDUVA':'339048','BUTIA':'339032','FREDERICO WESTPHALEN':'339048','NOVA ALVORADA':'339039','SOBRADINHO':'339039','QUARAI':'339030','CAMPINAS DO SUL':'339030','MIRAGUAI':'339030','PARAI':'339039','BROCHIER':'335043','BOM RETIRO DO SUL':'319016','HUMAITA':'339039','NOVO MACHADO':'339030','MATO QUEIMADO':'339039','SAO VALERIO DO SUL':'339039','TAPEJARA':'339039','TAPERA':'339091','LINDOLFO COLLOR':'339030','BOM JESUS':'469071','COLINAS':'339039','CRISSIUMAL':'339030','PANAMBI':'339030','SEDE NOVA':'339030','LAGOA BONITA DO SUL':'339030','TRES DE MAIO':'319016','TURUCU':'339032','ESTRELA':'469071','ALEGRETE':'319008','TEUTONIA':'335043','NOVA SANTA RITA':'339049','FORTALEZA DOS VALOS':'339039','MAXIMILIANO DE ALMEIDA':'339039','ROQUE GONZALES':'339032','PORTAO':'339030','BUTIA':'339030','COLORADO':'339039','NOVA PADUA':'339030','PIRAPO':'339030','NOVO TIRADENTES':'339030','PORTO XAVIER':'339030','ELDORADO DO SUL':'339032','ITAPUCA':'339030','IVORA':'339030','LAJEADO':'339030','SAO JOSE DO OURO':'339030','DOM FELICIANO':'469071','TRES DE MAIO':'319008','NICOLAU VERGUEIRO':'339030','FREDERICO WESTPHALEN':'339030','SANTANA DO LIVRAMENTO':'339014','VIAMAO':'339093','TAQUARA':'469171','ARATIBA':'335043','NOVO CABRAIS':'319016','BUTIA':'339049','NOVA ARACA':'335041','SAO BORJA':'319016','DOUTOR RICARDO':'339030','GIRUA':'319113','COTIPORA':'335043','MACAMBARA':'339030','CANOAS':'335041','CANOAS':'319091','SANTO ANGELO':'329022','TOROPI':'339030','LAJEADO':'335043','DERRUBADAS':'339030','JOIA':'319008','CHUI':'319016','LAJEADO':'319016','TRES CACHOEIRAS':'319016','RIO GRANDE':'329021','ALVORADA':'469071'}
#entre 2 e 4 desvios padrão:
#entre 4 ateh o final (21) desvios padrão:
exemplos = {'SAPUCAIA DO SUL':'469071','SAPIRANGA':'339093','PAROBE':'339093','FARROUPILHA':'339093','TAPEJARA':'469071','VISTA GAUCHA':'469071','BARRA DO QUARAI':'469071','SEBERI':'339091','CAMAQUA':'339091','NOVO HAMBURGO':'339047','NOVO HAMBURGO':'329021','ALEGRETE':'329021','JAGUARI':'335041','BARRACAO':'335041','PIRATINI':'319091','ALEGRETE':'319005','SAO VALERIO DO SUL':'319008','VALE REAL':'319046','GUAPORE':'335043','CANELA':'339093','BOSSOROCA':'319008','MAXIMILIANO DE ALMEIDA':'339032','CIRIACO':'319008','RIO GRANDE':'339093','CHAPADA':'449051','SEVERIANO DE ALMEIDA':'335043','SANTO CRISTO':'449051','SANTA MARIA':'319092','DOUTOR MAURICIO CARDOSO':'449051','GAURAMA':'449052','ERVAL SECO':'469071','RIO GRANDE':'469071','SAO JOSE DO OURO':'449052','SAPUCAIA DO SUL':'319091','CANOAS':'449052','ESTEIO':'319091','COTIPORA':'449051','JACUTINGA':'335041','CAPAO BONITO DO SUL':'449052','VERANOPOLIS':'449052','ESMERALDA':'449051','TIO HUGO':'449052','TRINDADE DO SUL':'339047','SAPIRANGA':'469071','MACHADINHO':'335043','LAJEADO':'319004','BOSSOROCA':'319113','ESTACAO':'319008','ELDORADO DO SUL':'449052','BALNEARIO PINHAL':'469071','CAPAO BONITO DO SUL':'339036','CARLOS BARBOSA':'469071','MATO LEITAO':'449052','SAPIRANGA':'329021','SAPIRANGA':'339091','CANDIDO GODOI':'339046','NOVO HAMBURGO':'333041','SAO BORJA':'339091','BOA VISTA DO BURICA':'449052','TRAMANDAI':'319004','COXILHA':'339032','TIRADENTES DO SUL':'335043','PASSO DO SOBRADO':'449052','NOVA HARTZ':'449051','PASSO FUNDO':'469071','MATO QUEIMADO':'339030','SAO JORGE':'339032','JULIO DE CASTILHOS':'319004','ENCRUZILHADA DO SUL':'335043','SALVADOR DAS MISSOES':'319004','CAMPO NOVO':'449051','SAO MIGUEL DAS MISSOES':'469071','TRES DE MAIO':'335043','GENTIL':'319004','IMIGRANTE':'335041','SANTO CRISTO':'339046','CANUDOS DO VALE':'449051','ESTEIO':'335043','SAO VALERIO DO SUL':'319113','CAMPO NOVO':'319004','CACHOEIRA DO SUL':'319004','SANTA CRUZ DO SUL':'315013','ALEGRETE':'339047','TUPARENDI':'319091','SAO PEDRO DO BUTIA':'339039','VERA CRUZ':'469071','RIO GRANDE':'319034','CHAPADA':'469071','NOVA PETROPOLIS':'335043','MORRO REUTER':'449052','IBIRAIARAS':'319016','SAO MARTINHO DA SERRA':'339032','NAO-ME-TOQUE':'449051','NOVO CABRAIS':'339039','TRES DE MAIO':'449052','BARAO DE COTEGIPE':'339030','SANTA ROSA':'469071','QUINZE DE NOVEMBRO':'449052','CONSTANTINA':'449052','SANTO ANTONIO DO PLANALTO':'319013','SANTA VITORIA DO PALMAR':'339030','SANTA TEREZA':'339039','VERA CRUZ':'329021'}

#plota os 30 primeiros exemplos:
contador = 0
for municipio in list(exemplos.keys())[0:30]:
	rubrica = exemplos[municipio]
	plt.figure(contador)
	ax = plotagem(projetado_modelo=False,bandas=True,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
	contador += 1

############

# Criacao das figuras da apresentacao do seminario Brasil Digital TCU-CGU setembro de 2018:
rubrica = '339047' #obrigacoes tributarias e contributivas
municipio = 'NOVO HAMBURGO'

# Figura 1:
#somente realizado do grupo:
plotagem(projetado_modelo=False,bandas=False,realizado=False,realizado_grupo=True,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)

#figura 2:
rubricas_grupos_projetadas_valor_original, r2_grupos = gera_projecao_via_linear_reg_trend_e_sazonalidade(rubricas_grupos)
#somente bandas:
ax = plotagem(projetado_modelo=False,bandas=True,realizado=False,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas_valor_original,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=True)
ax.set_ylabel('Valor em R$')
ax.set_ylim(1800000,8000000)
ax.set_title('')

#figura 3
rubricas_grupos_projetadas_somente_sazonalidade, r2_grupos_somente_sazonalidade = gera_projecao_via_linear_reg_somente_sazonalidade_para_apresentacao(rubricas)
#somente projetado do modelo:
ax = plotagem(projetado_modelo=False,bandas=False,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_grupos_projetadas_somente_sazonalidade,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
ax.set_ylim(-50000,2000000)
ax.set_title('')

#figura 4
ax = plotagem(projetado_modelo=True,bandas=True,realizado=False,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas_valor_original,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)
# pra incluir o realizado do grupo no mesmo grafico:
# plotagem(projetado_modelo=False,bandas=False,realizado=False,realizado_grupo=True,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)

#figura 5 e exemplos subsequentes (trocando municipio e rubrica):
#realizado do municipio + bandas
ax = plotagem(projetado_modelo=False,bandas=True,realizado=True,realizado_grupo=False,rubrica=rubrica,rubricas=rubricas_std,rubricas_grupos_projetadas=rubricas_grupos_projetadas,rubricas_grupos=rubricas_grupos,municipio=municipio,window=12,qtd_std_dev=2,grupos=grupos,only_mean=False)

#projecao em 2D da clusterizacao:
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

df_pca = pca.fit_transform(municipios_agrupados[[col for col in municipios_agrupados if 'grupo'not in col]])
df_pca = DataFrame(df_pca,index=municipios_agrupados.index,columns=['x','y'])

plt.plot(df_pca['x'],df_pca['y'], 'ro', alpha = 0.5)
for i in range(len(df_pca)):
    plt.text(df_pca['x'].iloc[i],df_pca['y'].iloc[i], str(df_pca.index[i]))
plt.show()

#tirando POA:
pca = PCA(n_components = 2)

scaler = StandardScaler()
municipios_agrupados_std = scaler.fit_transform(municipios_agrupados[[col for col in municipios_agrupados if 'grupo' not in col]])
# df_pca = pca.fit_transform(municipios_agrupados[[col for col in municipios_agrupados if 'grupo' not in col]].loc[[i for i in municipios_agrupados.index if 'PORTO ALEGRE' not in i]])
df_pca = pca.fit_transform(municipios_agrupados_std)
# df_pca = DataFrame(df_pca,index=[i for i in municipios_agrupados.index if 'PORTO ALEGRE' not in i],columns=['x','y'])
df_pca = DataFrame(df_pca,index=municipios_agrupados.index,columns=['x','y'])
df_pca['grupo'] = municipios_agrupados['grupo']

colors = {
	0:'b',
	1:'r',
	2:'g',
	3:'c',
	4:'m',
}

plt.scatter(df_pca['x'],df_pca['y'],color=df_pca['grupo'].map(colors))


df_pca = df_pca.loc[[i for i in df_pca.index if 'PORTO ALEGRE' not in i]]


plt.scatter(df_pca['x'],df_pca['y'],color=df_pca['grupo'].map(colors))

for municipio in df_pca.index:
	if df_pca['grupo'].loc[municipio] == 3:
    	plt.text(df_pca['x'].loc[municipio],df_pca['y'].loc[municipio], str(municipio),fontsize=8)

############
# FIM
############
