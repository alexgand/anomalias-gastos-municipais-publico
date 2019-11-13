# %matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift
import pickle
import os
from sklearn.linear_model import LinearRegression
import unicodedata
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#seaborn plotting:
import seaborn as sns
from tqdm import tqdm

def strip_accents(s):
	#retira acentos de strings
	s = s.replace('`','').replace("'",'')
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def arruma_nomes_das_colunas(base):
	#arruma nomes das keys pra ficar soh com o nome do municipio:
	keys_originais = list(base.keys())
	for key in keys_originais:
		new_key = key.replace('PM DE ','').replace('CM DE ','').replace('.csv','').replace('.xlsx','')
		new_key = strip_accents(new_key) #retira acentos
		# base[new_key] = base.pop(key)
		base[new_key] = base[key].copy()
		del base[key]
	return base

def limpa_base(base):
	#fica somente poder executivo, sai RPP e CMM:
	for municipio in base:
		base[municipio] = base[municipio][base[municipio]['PODER'] == 'E']
		base[municipio] = base[municipio][base[municipio]['TP_UNIDADE'] != 'RPP']
		base[municipio] = base[municipio][base[municipio]['TP_UNIDADE'] != 'CMM']
		print(municipio)
		if not isinstance(base[municipio]['DT_LIQUIDACAO'].iloc[0], pd._libs.tslib.Timestamp):
			base[municipio]['DT_LIQUIDACAO'] = pd.to_datetime(base[municipio]['DT_LIQUIDACAO'],dayfirst=True) #se nao estiver formatado como data (checa o primeiro registro apenas), converte agora.
	return base

def grava(coisa,filename,path):
	pkl_file = open(path + filename, 'wb')
	pickle.dump(coisa, pkl_file)
	pkl_file.close()

def abre(filename,path):
	#formato do path:'/Users/Alexandre/'
	pkl_file = open(path + filename, 'rb')
	coisa = pickle.load(pkl_file)
	pkl_file.close()
	return coisa

def descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.xlsx'):
	#Descobre arquivos na pasta:
	arquivos = []
	for file in os.listdir(pasta):
		arquivos.append(os.fsdecode(file))
	arquivos = [arquivo for arquivo in arquivos if tipo_do_arquivo in arquivo] #seleciona soh arquivos com .xlsx
	return arquivos

def cria_base_de_municipios_a_partir_de_xlsx_ou_csv(pasta,arquivos):
	#cria base de dados e le arquivos
	base={}
	colunas_a_manter = ['NM_ORGAO','NM_UNIDADE','TP_UNIDADE','NR_LIQUIDACAO','DT_LIQUIDACAO','VL_LIQUIDACAO','NR_EMPENHO','ANO_EMPENHO','CD_CREDOR','CD_CONTA_SG','PODER','RESTOS_PAGAR']
	for arquivo in arquivos:
		if '.xlsx' in arquivo:
			base[arquivo] = pd.read_excel(pasta+arquivo,dayfirst=True)
		elif '.csv' in arquivo:
			with open(pasta+arquivo, 'r') as fp: #necessario por problemas no encoding das letras com acentos.
				base[arquivo] = pd.read_csv(fp, sep = ";", decimal = ",", encoding = "utf-8",keep_date_col=True,parse_dates=['DT_LIQUIDACAO'],dayfirst=True) #jah transforma data em data.
		base[arquivo] = base[arquivo][colunas_a_manter]
		print('Arquivo',arquivo,'importado com sucesso.')
	return base

def gera_series_agrupadas(base,coluna_a_manter = 'VL_LIQUIDACAO',agrupador1 = 'DT_LIQUIDACAO',agrupador2 = 'rubrica'):
	series_rubricas = {}
	for municipio in base:
		df_temp = base[municipio].groupby([base[municipio][agrupador1],base[municipio][agrupador2]]).sum()
		df_temp = df_temp[coluna_a_manter] #mantem soh coluna do valor
		df_temp = df_temp.unstack()
		df_temp = df_temp.groupby([pd.TimeGrouper('M')]).sum() #agrega por mes
		series_rubricas[municipio] = df_temp
	return series_rubricas

def cria_grupos_de_municipios_parecidos_via_k_means(pasta,arquivo,qtd_grupos,random_seed):
	#agrupamento dos municipios parecidos:
	# arquivo = 'X:/DCF/SAICE/SAE-I/pub/Piloto análise de dados/PIB_RS_2015-alterado.xlsx'
	municipios = pd.read_excel(pasta+arquivo)
	municipios = municipios.set_index('nm_munic')

	# municipios = pd.get_dummies(municipios) #one hot encoding
	municipios = municipios.select_dtypes(['number']) #somente colunas numericas
	del municipios['PIB_per_capita'] #retira PIB_per_capita, que eh redundante

	scaler = StandardScaler()
	municipios_std = scaler.fit_transform(municipios)

	agrupamentos = KMeans(n_clusters=qtd_grupos,random_state=random_seed,n_jobs=-1)
	# agrupamentos = MeanShift(n_jobs=-1) #sem pre-estabelecer quantidade de grupos.

	model = agrupamentos.fit(municipios_std)

	municipios_agrupados = municipios.copy()
	municipios_agrupados['grupo'] = model.labels_

	print('Grupos gerados:')
	print(sorted(municipios_agrupados['grupo'].unique()))
	print('5 primeiros municipios de cada grupo:')
	grupos = {}
	for grupo_unique in sorted(municipios_agrupados['grupo'].unique()):
		print('Grupo',grupo_unique,':')
		print(list(municipios_agrupados[municipios_agrupados.grupo == grupo_unique].index))
		grupos[grupo_unique] = list(municipios_agrupados[municipios_agrupados.grupo == grupo_unique].index)
		print('')

	return municipios_agrupados, grupos

def cria_grupo_agrupado_das_rubricas_comuns_a_todos_os_municipios_parecidos(rubricas,grupos,retirar_meses_com_zeros=False):
	#soma todas as rubricas dos municipios do grupo. Se nao ha a rubrica pra algum municipio, considera valor como zero.
	rubricas_grupos = {}
	for grupo in tqdm(grupos):
		contador = 0
		for municipio in tqdm(rubricas):
			if municipio in grupos[grupo]:
				if contador == 0:
					soma_do_grupo = rubricas[municipio].copy()
				else:
					# soma_do_grupo = soma_do_grupo + rubricas[municipio]
					soma_do_grupo = soma_do_grupo.add(rubricas[municipio], fill_value=0) #soma os dataframes, equivalente a +, mas nao gera nans quando nao ha a rubrica em um dos municipios, considera como zero.
				contador +=1
				soma_do_grupo = soma_do_grupo.dropna(axis=1) #tira as colunas com nans
				if retirar_meses_com_zeros:
					soma_do_grupo = soma_do_grupo.replace(0,np.nan).dropna(axis=1)
		if contador > 0:
			rubricas_grupos[grupo] = soma_do_grupo
			del soma_do_grupo
	return rubricas_grupos

def standardizacao(dict_of_dfs):
	std = {}
	for key in tqdm(dict_of_dfs):
		scaler = StandardScaler()
		#se existiu rubrica que passou dos filtros (pgto zero, etc.)
		if len(dict_of_dfs[key].columns) > 0:
			std[key] = DataFrame(scaler.fit_transform(dict_of_dfs[key]),index=dict_of_dfs[key].index,columns=dict_of_dfs[key].columns)
		else:
			print(key,'nao teve rubricas que passaram nos filtros (pgto zero, etc.)')
	return std

def cria_bandas(data, window, qtd_std_dev):
	rolling_mean = data.rolling(window).mean()
	rolling_std = data.rolling(window).std()
	bandas = DataFrame()
	bandas['mean'] = rolling_mean
	bandas['high'] = rolling_mean + (rolling_std * qtd_std_dev)
	bandas['low'] = rolling_mean - (rolling_std * qtd_std_dev)
	bandas['std'] = rolling_std
	return bandas

def gera_projecao_via_linear_reg_trend_e_sazonalidade(rubricas_std):
	rubricas_projetadas = {}
	r2 = {}
	for municipio in tqdm(rubricas_std):
		model = LinearRegression(n_jobs=-1)
		X = DataFrame(index=rubricas_std[municipio].index)
		X['mes'] = X.index.month
		X = pd.get_dummies(X['mes'])
		tempo = np.array(range(len(rubricas_std[municipio]))).reshape(-1,1)
		X['tempo'] = tempo
		model.fit(X,rubricas_std[municipio])
		pred = model.predict(X)
		pred = DataFrame(pred,index=rubricas_std[municipio].index,columns=rubricas_std[municipio].columns)
		rubricas_projetadas[municipio] = pred
		temp = {}
		for rubrica in rubricas_projetadas[municipio]:
			temp[rubrica] = r2_score(rubricas_std[municipio][rubrica],rubricas_projetadas[municipio][rubrica])
		r2[municipio] = temp
	return rubricas_projetadas, r2

def gera_projecao_via_linear_reg_somente_sazonalidade_para_apresentacao(rubricas_std):
	rubricas_projetadas = {}
	r2 = {}
	for municipio in rubricas_std:
		model = LinearRegression(n_jobs=-1)
		X = DataFrame(index=rubricas_std[municipio].index)
		X['mes'] = X.index.month
		X = pd.get_dummies(X['mes'])
		# tempo = np.array(range(len(rubricas_std[municipio]))).reshape(-1,1)
		# X['tempo'] = tempo
		model.fit(X,rubricas_std[municipio])
		pred = model.predict(X)
		pred = DataFrame(pred,index=rubricas_std[municipio].index,columns=rubricas_std[municipio].columns)
		rubricas_projetadas[municipio] = pred
		temp = {}
		for rubrica in rubricas_projetadas[municipio]:
			temp[rubrica] = r2_score(rubricas_std[municipio][rubrica],rubricas_projetadas[municipio][rubrica])
		r2[municipio] = temp
	return rubricas_projetadas, r2

def em_comparacao_com_o_grupo_encontra_rubricas_fora_das_bandas(grupos,rubricas_std,rubricas_grupos_projetadas,window,qtd_std_dev,qtd_meses_atras,rubricas):
	# anomalias_comparando_com_grupo = DataFrame(columns=['grupo','municipio','rubrica','high/low','mes','valor_total_rubrica','%_diferenca','qtd_desvios'])
	anomalias_comparando_com_grupo = DataFrame(columns=['grupo','municipio','rubrica','mes','valor_total_rubrica','qtd_desvios'])
	for grupo in tqdm(grupos):
		# print('Grupo:',grupo)
		for municipio in tqdm(grupos[grupo]):
			if municipio in rubricas_std:
				# print('Municipio',municipio)
				for rubrica in rubricas_grupos_projetadas[grupo]:
					if rubrica in rubricas_std[municipio].columns: #se a rubrica existir no municipio:
						bandas = cria_bandas(rubricas_grupos_projetadas[grupo][rubrica],window=window,qtd_std_dev=qtd_std_dev)
						for mes in (np.array(range(qtd_meses_atras+1)[:0:-1])*-1): #gera lista -5,-4,-3,-2,-1, por exemplo.
							if ( (rubricas_std[municipio][rubrica].iloc[mes] > bandas['high'].iloc[mes]) and (rubricas_std[municipio][rubrica].iloc[mes] > rubricas_grupos_projetadas[grupo][rubrica].iloc[mes]) ) or ( (rubricas_std[municipio][rubrica].iloc[mes] < bandas['low'].iloc[mes]) and (rubricas_std[municipio][rubrica].iloc[mes] < rubricas_grupos_projetadas[grupo][rubrica].iloc[mes]) ):
								linha_a_adicionar = [grupo,municipio,rubrica,rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes],(rubricas_std[municipio][rubrica].iloc[mes] - rubricas_grupos_projetadas[grupo][rubrica].iloc[mes]) / bandas['std'].iloc[mes] ]
								if len(anomalias_comparando_com_grupo.index) > 0: #se jah tem algum registro:
									anomalias_comparando_com_grupo.loc[anomalias_comparando_com_grupo.index[-1]+1] = linha_a_adicionar
								else:
									anomalias_comparando_com_grupo.loc[0] = linha_a_adicionar

							# if rubricas_std[municipio][rubrica].iloc[mes] > bandas['high'].iloc[mes]:
							# 	if len(anomalias_comparando_com_grupo.index) > 0: #se jah tem algum registro:
							# 		anomalias_comparando_com_grupo.loc[anomalias_comparando_com_grupo.index[-1]+1] = [grupo,municipio,rubrica,'high',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #adiciona nova linha
							# 	else:
							# 		anomalias_comparando_com_grupo.loc[0] = [grupo,municipio,rubrica,'high',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #primeiro registro
							# 	# print(rubrica,'high')
							# elif rubricas_std[municipio][rubrica].iloc[mes] < bandas['low'].iloc[mes]:
							# 	if len(anomalias_comparando_com_grupo.index) > 0: #se jah tem algum registro:
							# 		anomalias_comparando_com_grupo.loc[anomalias_comparando_com_grupo.index[-1]+1] = [grupo,municipio,rubrica,'low',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #adiciona nova linha
							# 	else:
							# 		anomalias_comparando_com_grupo.loc[0] = [grupo,municipio,rubrica,'low',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #primeiro registro
								# print(rubrica,'low')
	return anomalias_comparando_com_grupo
		
def em_comparacao_com_o_proprio_municipio_encontra_rubricas_fora_das_bandas(grupos,rubricas_std,rubricas_projetadas,window,qtd_std_dev,qtd_meses_atras):
	anomalias_comparando_com_o_proprio_municipio = DataFrame(columns=['grupo','municipio','rubrica','high/low','mes','valor_total_rubrica'])
	for municipio in tqdm(rubricas_std):
		# print('Municipio',municipio)
		grupo = [key for key in grupos if municipio in grupos[key]][0]
		for rubrica in rubricas_std[municipio]:
			bandas = cria_bandas(rubricas_projetadas[municipio][rubrica],window=window,qtd_std_dev=qtd_std_dev)
			for mes in (np.array(range(qtd_meses_atras+1)[:0:-1])*-1): #gera lista -5,-4,-3,-2,-1, por exemplo.
				if rubricas_std[municipio][rubrica].iloc[mes] > bandas['high'].iloc[mes]:
					if len(anomalias_comparando_com_o_proprio_municipio.index) > 0: #se jah tem algum registro:
						anomalias_comparando_com_o_proprio_municipio.loc[anomalias_comparando_com_o_proprio_municipio.index[-1]+1] = [grupo,municipio,rubrica,'high',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #adiciona nova linha
					else:
						anomalias_comparando_com_o_proprio_municipio.loc[0] = [grupo,municipio,rubrica,'high',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #primeiro registro
					# print(rubrica,'high')
				elif rubricas_std[municipio][rubrica].iloc[mes] < bandas['low'].iloc[mes]:
					if len(anomalias_comparando_com_o_proprio_municipio.index) > 0: #se jah tem algum registro:
						anomalias_comparando_com_o_proprio_municipio.loc[anomalias_comparando_com_o_proprio_municipio.index[-1]+1] = [grupo,municipio,rubrica,'low',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #adiciona nova linha
					else:
						anomalias_comparando_com_o_proprio_municipio.loc[0] = [grupo,municipio,rubrica,'low',rubricas_std[municipio][rubrica].index[mes],rubricas[municipio][rubrica].iloc[mes]] #primeiro registro				
					# print(rubrica,'low')
	return anomalias_comparando_com_o_proprio_municipio

def mostra_lancamentos_das_rubricas_anormais_somente_no_ultimo_mes(anomalias,base,qtd_lancamentos):
	#Se informar qtd_lancamentos = 0, mostra todos.
	lancamentos = {}
	for municipio in tqdm(anomalias['municipio'].unique()):
		lancamentos[municipio] = DataFrame(columns = base[municipio].columns)
		for rubrica in anomalias[anomalias['municipio'] == municipio]['rubrica'].unique():
			lancamentos[municipio] = pd.concat([lancamentos[municipio],base[municipio][base[municipio]['rubrica'] == rubrica]])
		ultimo_ano = max(lancamentos[municipio]['DT_LIQUIDACAO'].dt.year)
		lancamentos[municipio] = lancamentos[municipio][lancamentos[municipio]['DT_LIQUIDACAO'].dt.year == ultimo_ano]
		ultimo_mes = max(lancamentos[municipio]['DT_LIQUIDACAO'].dt.month)
		lancamentos[municipio] = lancamentos[municipio][lancamentos[municipio]['DT_LIQUIDACAO'].dt.month == ultimo_mes]
		lancamentos[municipio] = lancamentos[municipio].sort_values(['rubrica','VL_LIQUIDACAO'],ascending=False)
		if qtd_lancamentos > 0:
			lancamentos[municipio] = lancamentos[municipio].groupby('rubrica').head(qtd_lancamentos) #soh os primeiros maiores de cada rubrica
	return lancamentos

def descobre_erros_no_mes_final(rubricas):
	ultimo_mes=DataFrame(columns=['ultimo_mes','municipio'])
	contador = 0
	for municipio in rubricas:
		ultimo_mes.loc[contador] = [rubricas[municipio].index[-1], municipio]
		contador += 1
	print('Contagem de municipios com mes final:')
	print(ultimo_mes.groupby('ultimo_mes').count())
	#seleciona para analise o mes que tem a maior parte dos municipios:
	mes_da_analise = ultimo_mes.groupby('ultimo_mes').count().sort_values(by='municipio').index[-1]
	print('')
	print('Orgaos com problemas - provavelmente ainda nao enviaram a ultima remessa:')
	print(ultimo_mes[ultimo_mes['ultimo_mes'] != mes_da_analise])
	problemas = list(ultimo_mes[ultimo_mes['ultimo_mes'] != mes_da_analise]['municipio'])
	return ultimo_mes, problemas, mes_da_analise

def define_mes_inicial(rubricas, mes_inicial):
	for municipio in rubricas:
		rubricas[municipio] = rubricas[municipio][mes_inicial:]
	return rubricas

def acha_grupo_do_municipio(municipio_a_encontrar,grupos):
	for grupo in grupos:
		if municipio_a_encontrar in grupos[grupo]:
			return grupo

def ordena_series_pelos_maiores_valores(rubricas):
	maiores_rubricas = DataFrame()
	for municipio in rubricas:
		if len(maiores_rubricas) == 0: #primeiro run:
			maiores_rubricas = DataFrame(rubricas[municipio].max().sort_values(ascending=False),columns=[municipio])
		else:
			df2 = DataFrame(rubricas[municipio].max().sort_values(ascending=False),columns=[municipio])
			maiores_rubricas = pd.concat([maiores_rubricas, df2], axis=1)
	return maiores_rubricas

def plot_rubrica_ordenadas_pelas_maiores_do_municipio(maiores_rubricas,rubricas,rubricas_grupos_projetadas,rubricas_projetadas,municipio,ordem,window,qtd_std_dev):
	rubrica = maiores_rubricas[municipio].sort_values(ascending=False).index[ordem-1]
	if rubricas_grupos_projetadas:
		grupo = acha_grupo_do_municipio(municipio_a_encontrar=municipio)
		cria_bandas(rubricas_grupos_projetadas[grupo][rubrica],window,qtd_std_dev).plot()
	if rubricas_projetadas:
		cria_bandas(rubricas_projetadas[municipio][rubrica],window,qtd_std_dev).plot()
		pass
	if rubricas:
		rubricas[municipio][rubrica].plot()

def gera_series_agrupadas_a_partir_de_arquivos(arquivos,pasta,poder,pasta_a_gravar_CM,qtd_digitos_rubrica,coluna_a_manter = 'VL_LIQUIDACAO',agrupador1 = 'DT_LIQUIDACAO',agrupador2 = 'rubrica'):
	series_rubricas = {}
	contador = 0
	for arquivo in tqdm(arquivos):
		df_temp = abre(arquivo,pasta)
		nome_do_orgao = df_temp['NOME'].iloc[0]
		#adiciona rubrica:
		df_temp['CD_CONTA_SG'] = df_temp['CD_CONTA_SG'].astype(str)
		# df_temp['rubrica'] = df_temp['CD_CONTA_SG'].str[:6]
		df_temp['rubrica'] = df_temp['CD_CONTA_SG'].str[:qtd_digitos_rubrica]
		#limpa:
		if poder == 'PM':
			#novo dataframe pra gravar dados da CM, se vier junto com o arquivo da PM:
			df_cmm = df_temp[df_temp['PODER'] == 'L']
			df_cmm = df_cmm[df_cmm['TP_UNIDADE'] == 'CMM']

			df_temp = df_temp[df_temp['PODER'] == 'E']
			df_temp = df_temp[df_temp['TP_UNIDADE'] != 'CMM']
		elif poder == 'CM':
			df_cmm = []
			
			df_temp = df_temp[df_temp['PODER'] == 'L']
		
		df_temp = df_temp[df_temp['TP_UNIDADE'] != 'RPP']
		if not isinstance(df_temp['DT_LIQUIDACAO'].iloc[0], pd._libs.tslib.Timestamp):
			df_temp['DT_LIQUIDACAO'] = pd.to_datetime(df_temp['DT_LIQUIDACAO'],dayfirst=True) #se nao estiver formatado como data (checa o primeiro registro apenas), converte agora.
		#agrupa:
		df_temp = df_temp.groupby([df_temp[agrupador1],df_temp[agrupador2]]).sum()
		df_temp = df_temp[coluna_a_manter] #mantem soh coluna do valor
		df_temp = df_temp.unstack()
		# df_temp = df_temp.groupby([pd.TimeGrouper('M')]).sum() #agrega por mes #old pandas
		df_temp = df_temp.groupby([pd.Grouper(freq='M')]).sum() #agrega por mes
		series_rubricas[nome_do_orgao] = df_temp #pega o primeiro nome do orgao
		# print(nome_do_orgao,'importado OK. Numero',contador,'de um total de',len(arquivos))
		contador+=1

		#grava os arquivos das CMs:
		if len(df_cmm) > 0:
			nome_do_orgao = nome_do_orgao.replace('PM','CM')
			grava(coisa=df_cmm,filename=nome_do_orgao+'.pkl',path=pasta_a_gravar_CM)

	return series_rubricas

def mostra_lancamentos_das_rubricas_anormais(anomalias,pasta,arquivos,qtd_lancamentos,qtd_meses_atras,mes_da_analise,poder,qtd_digitos_rubrica):
	#Se informar qtd_lancamentos = 0, mostra todos.
	lancamentos = {}
	erros = []
	contador = 0
	for arquivo in tqdm(arquivos):
		df_temp = abre(arquivo,pasta)
		municipio = df_temp['NOME'].iloc[0].replace('PM DE ','').replace('CM DE ','').replace('.csv','').replace('.xlsx','').replace('.pkl','').replace('CM DE ','').replace('OU DE ','')
		municipio = strip_accents(municipio) #retira acentos

		if municipio not in anomalias['municipio'].unique():
			print('Orgao',municipio,'foi excluido, nao enviou remessa ou nao teve anomalia no periodo.')
			print('')
			erros.append(municipio)

		else:
			# print('Orgao',municipio,'inicio processamento.')		
		
			#adiciona rubrica:
			df_temp['CD_CONTA_SG'] = df_temp['CD_CONTA_SG'].astype(str)
			df_temp['rubrica'] = df_temp['CD_CONTA_SG'].str[:qtd_digitos_rubrica]
		
			if poder == 'PM':
				#limpa:
				df_temp = df_temp[df_temp['PODER'] == 'E']
				df_temp = df_temp[df_temp['TP_UNIDADE'] != 'RPP']
				df_temp = df_temp[df_temp['TP_UNIDADE'] != 'CMM']
			
			elif poder == 'CM':
				df_temp = df_temp[df_temp['PODER'] == 'L']
				df_temp = df_temp[df_temp['TP_UNIDADE'] != 'RPP']
		
			if not isinstance(df_temp['DT_LIQUIDACAO'].iloc[0], pd._libs.tslib.Timestamp):
				df_temp['DT_LIQUIDACAO'] = pd.to_datetime(df_temp['DT_LIQUIDACAO'],dayfirst=True) #se nao estiver formatado como data (checa o primeiro registro apenas), converte agora.

			lancamentos[municipio] = DataFrame(columns = df_temp.columns)
			# for rubrica, mes in anomalias[anomalias['municipio'] == municipio]['rubrica'].unique():
			pares = anomalias[anomalias['municipio'] == municipio][['rubrica','mes']].apply(tuple,axis=1)
			# for rubrica, mes in anomalias[anomalias['municipio'] == municipio]['rubrica'].unique():
			for rubrica, mes in pares:
				lancamentos[municipio] = pd.concat([lancamentos[municipio],df_temp[(df_temp['rubrica'] == rubrica) & (df_temp['DT_LIQUIDACAO'].apply(lambda x: x.month) == mes.month) & (df_temp['DT_LIQUIDACAO'].apply(lambda x: x.year) == mes.year) ]])

			ultimo_mes = max(lancamentos[municipio]['DT_LIQUIDACAO'])
			#coloca o dia 1 do mes pra comecar, estava comecando no dia 31, aih deixava lancamentos de fora.
			mes_a_comecar = (ultimo_mes - pd.Timedelta(qtd_meses_atras,'M')).replace(day=1)
			# mes_a_comecar = (maior_mes - pd.Timedelta(qtd_meses_atras,'M'))
			# lancamentos[municipio] = lancamentos[municipio].set_index('DT_LIQUIDACAO')
			lancamentos[municipio]['ANOMES'] = lancamentos[municipio]['DT_LIQUIDACAO'].map(lambda x: 100*x.year + x.month)
			lancamentos[municipio] = lancamentos[municipio][lancamentos[municipio]['DT_LIQUIDACAO'] >= mes_a_comecar]
			lancamentos[municipio] = lancamentos[municipio].sort_values(['rubrica','ANOMES','VL_LIQUIDACAO'],ascending=False)

			# print('Orgao',municipio,'processado OK.',contador,'de um total de',len(arquivos))
			# print('')
			contador+=1
			
			if qtd_lancamentos > 0:
				lancamentos[municipio] = lancamentos[municipio].groupby(['rubrica','ANOMES']).head(qtd_lancamentos) #soh os primeiros maiores de cada rubrica
				# lancamentos[municipio] = lancamentos[municipio].groupby(['rubrica','ANOMES']).sum()['VL_LIQUIDACAO'].unstack().T #df, rubricas nas colunas, meses nas linhas.
				# lancamentos[municipio]['cumsum'] = lancamentos[municipio].groupby('rubrica').sum().groupby('ANOMES').cumsum()

				# lancamentos[municipio]['cumsum'] = 0
				# lancamentos[municipio]['cumsum'] = np.where(lancamentos[municipio]['ANOMES'] != lancamentos[municipio]['ANOMES'].shift(), 0, lancamentos[municipio]['cumsum'].shift() + lancamentos[municipio]['VL_LIQUIDACAO'] )
				# lancamentos[municipio]['cumsum'] = np.where(lancamentos[municipio]['ANOMES'] != lancamentos[municipio]['ANOMES'].shift(), 0, lancamentos[municipio]['VL_LIQUIDACAO'].cumsum() )
				# lancamentos[municipio]['sum'] = lancamentos[municipio].groupby(['rubrica','ANOMES']).sum()['VL_LIQUIDACAO']
				# lancamentos[municipio]['sum'] = lancamentos['ALEGRETE'].groupby(['rubrica','ANOMES']).sum()['VL_LIQUIDACAO']
				# lancamentos[municipio]['percent'] = lancamentos[municipio]['cumsum'] / lancamentos[municipio]['sum']
				# lancamentos[municipio] = lancamentos[municipio][lancamentos[municipio]['percent'] < qtd_lancamentos]

	return lancamentos, erros

def retira_rubricas_com_muitos_meses_sem_pagamento(rubricas,percentual_maximo_de_meses_com_zero_pagamento):
#retira rubricas com muitos meses com zero
	for municipio in tqdm(rubricas):
		rubricas_originais = list(rubricas[municipio].columns)
		for rubrica in rubricas_originais:
			if (1 - (np.count_nonzero(rubricas[municipio][rubrica])/len(rubricas[municipio][rubrica])) ) > percentual_maximo_de_meses_com_zero_pagamento:
				# print('Municipio',municipio,'Rubrica',rubrica,'excluida, percentual de meses com zero pagamento:', 100 * (1 - (np.count_nonzero(rubricas[municipio][rubrica])/len(rubricas[municipio][rubrica])) ), '%' )
				del rubricas[municipio][rubrica]
	return rubricas

def retira_rubricas_com_muitos_meses_sem_pagamento_nos_ultimos_meses(rubricas,qtd_meses_atras,percentual_maximo_de_meses_com_zero_pagamento):
#retira rubricas com muitos meses com zero
	for municipio in tqdm(rubricas):
		rubricas_originais = list(rubricas[municipio].columns)
		for rubrica in rubricas_originais:
			if (1 - (np.count_nonzero(rubricas[municipio][rubrica][-qtd_meses_atras:])/len(rubricas[municipio][rubrica][-qtd_meses_atras:])) ) > percentual_maximo_de_meses_com_zero_pagamento:
				# print('Municipio',municipio,'Rubrica',rubrica,'excluida, percentual de meses com zero pagamento:', 100 * (1 - (np.count_nonzero(rubricas[municipio][rubrica][-qtd_meses_atras:])/len(rubricas[municipio][rubrica][-qtd_meses_atras:])) ), '%' )
				del rubricas[municipio][rubrica]
	return rubricas

def retira_rubricas_a_partir_da_numeracao_inicial(rubricas,inicial):
#pra tirar rubricas de despesas de capital: coomecam com '4'
	for municipio in tqdm(rubricas):
		rubricas_originais = list(rubricas[municipio].columns)
		for rubrica in rubricas_originais:
			if rubrica.startswith(inicial):
				# print('Municipio',municipio,'Rubrica',rubrica,'excluida.')
				del rubricas[municipio][rubrica]
	return rubricas

def retira_rubricas(rubricas,lista_de_rubricas_a_deletar):
#pra tirar folha de pagamento (319011), por exemplo:
	for municipio in tqdm(rubricas):
		rubricas_originais = list(rubricas[municipio].columns)
		for rubrica in rubricas_originais:
			if ( (rubrica in lista_de_rubricas_a_deletar) and (rubrica in rubricas[municipio]) ):
				# print('Municipio',municipio,'Rubrica',rubrica,'excluida.')
				del rubricas[municipio][rubrica]
	return rubricas

def gera_arquivo_exportacao_csv(lancamentos,pasta):
	for municipio in tqdm(lancamentos):
		lancamentos[municipio].to_csv(pasta+municipio+'.csv',decimal=',',sep=';',encoding='utf-8-sig')

# def plotagem(rubrica,rubricas,rubricas_grupos_projetadas,rubricas_projetadas,municipio,window,qtd_std_dev):
def plotagem(projetado_modelo,bandas,realizado,realizado_grupo,rubrica,rubricas,rubricas_grupos_projetadas,rubricas_grupos,municipio,window,qtd_std_dev,grupos,only_mean):
	# sns.set(style="dark")
	sns.set_style("dark", {"xtick.major.size": 6,"ytick.major.size": 6})
	# sns.set_style("ticks", {"axes.facecolor": '#EAEAF2'})
	
	resultado = adiciona_significado_da_rubrica_ate_elemento_somente_com_rubrica(rubrica=rubrica,arquivo='./data/Tabelas Auxiliares.xlsx')
	grupo = acha_grupo_do_municipio(municipio_a_encontrar=municipio,grupos=grupos)
	
	if bandas:
		df = cria_bandas(rubricas_grupos_projetadas[grupo][rubrica],window,qtd_std_dev)
		filtro = [col for col in df if 'std' not in col]
		if only_mean:
			filtro = [col for col in df if 'mean' in col]
		ax = df[filtro].plot(legend=True,figsize = (8,6))
		title = 'Natureza ' + rubrica + ', Município: ' + municipio + '\n' + str(resultado[0:2]) + '\n' + str(resultado[2]) + '\n' + str(resultado[3])	

	if projetado_modelo:
		ax = rubricas_grupos_projetadas[grupo][rubrica].plot(legend=True,figsize = (8,6))
		title = 'Natureza ' + rubrica + ', Município: ' + municipio + '\n' + str(resultado[0:2]) + '\n' + str(resultado[2]) + '\n' + str(resultado[3])
	# if rubricas_projetadas:
	# 	df = cria_bandas(rubricas_projetadas[municipio][rubrica],window,qtd_std_dev)
	# 	filtro = [col for col in df if 'std' not in col]
	# 	df[filtro].plot(title='Natureza ' + rubrica + ', Município: ' + municipio)
	if realizado:
		ax = rubricas[municipio][rubrica].plot(legend=True,figsize = (8,6))
		title = 'Natureza ' + rubrica + ', Município: ' + municipio + '\n' + str(resultado[0:2]) + '\n' + str(resultado[2]) + '\n' + str(resultado[3])

	if realizado_grupo:
		ax = rubricas_grupos[grupo][rubrica].plot(legend=True,figsize = (8,6))
		title = 'Natureza ' + rubrica + ', Grupo: ' + str(grupo) + '\n' + str(resultado[0:2]) + '\n' + str(resultado[2]) + '\n' + str(resultado[3])
		ax.set_ylabel("Valor agregado em R$")

	ax.set_title(title,fontsize=10)
	ax.set_xlabel("")

	return ax

def adiciona_significado_da_rubrica_ate_elemento(lancamentos,arquivo):
	significado = pd.read_excel(arquivo,sheet_name=None)
	for sheet in tqdm(significado.keys()):
		df = significado[sheet].set_index(significado[sheet].columns[0]) #label da primeira coluna como index
		contador = 0
		for municipio in tqdm(lancamentos):
			if sheet == 'Categoria Economica': #primeiro digito
				# lancamentos[municipio][sheet] = lancamentos[municipio]['rubrica'].apply(lambda x: df.loc[int(x[0])][0])
				get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[0])][0])
			if sheet == 'Natureza Despesa':
				# lancamentos[municipio][sheet] = lancamentos[municipio]['rubrica'].apply(lambda x: df.loc[int(x[1])][0])
				get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[1])][0])
			if sheet == 'Modalidade Aplicação':
				# lancamentos[municipio][sheet] = lancamentos[municipio]['rubrica'].apply(lambda x: df.loc[int(x[2:4])][0])
				get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[2:4])][0])
			if sheet == 'Elemento Despesa':
				# lancamentos[municipio][sheet] = lancamentos[municipio]['rubrica'].apply(lambda x: df.loc[int(x[4:6])][0])
				get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[4:6])][0])
			lancamentos[municipio][sheet] = get_categoria_vect(lancamentos[municipio]['rubrica'])
			# print('Municipio',municipio,'processado',contador,'de um total de',len(lancamentos.keys()))
			contador += 1
		# print('')
		# print('Sheet',sheet,'processada.')
		# print('')
	return lancamentos

def adiciona_significado_da_rubrica_ate_elemento_nas_anomalias(anomalias,arquivo):
	significado = pd.read_excel(arquivo,sheet_name=None)
	for sheet in tqdm(significado.keys()):
		df = significado[sheet].set_index(significado[sheet].columns[0]) #label da primeira coluna como index
		if sheet == 'Categoria Economica': #primeiro digito
			# anomalias[municipio][sheet] = anomalias[municipio]['rubrica'].apply(lambda x: df.loc[int(x[0])][0])
			get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[0])][0])
		if sheet == 'Natureza Despesa':
			# anomalias[municipio][sheet] = anomalias[municipio]['rubrica'].apply(lambda x: df.loc[int(x[1])][0])
			get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[1])][0])
		if sheet == 'Modalidade Aplicação':
			# anomalias[municipio][sheet] = anomalias[municipio]['rubrica'].apply(lambda x: df.loc[int(x[2:4])][0])
			get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[2:4])][0])
		if sheet == 'Elemento Despesa':
			# anomalias[municipio][sheet] = anomalias[municipio]['rubrica'].apply(lambda x: df.loc[int(x[4:6])][0])
			get_categoria_vect = np.vectorize(lambda x: df.loc[int(x[4:6])][0])
		anomalias[sheet] = get_categoria_vect(anomalias['rubrica'])
	return anomalias

def adiciona_significado_da_rubrica_ate_elemento_somente_com_rubrica(rubrica,arquivo):
	significado = pd.read_excel(arquivo,sheet_name=None)
	resultado = []
	for sheet in tqdm(significado.keys()):
		df = significado[sheet].set_index(significado[sheet].columns[0]) #label da primeira coluna como index
		if sheet == 'Categoria Economica': #primeiro digito
			resultado.append(df.loc[int(rubrica[0])][0])
		if sheet == 'Natureza Despesa':
			resultado.append(df.loc[int(rubrica[1])][0])
		if sheet == 'Modalidade Aplicação':
			resultado.append(df.loc[int(rubrica[2:4])][0])
		if sheet == 'Elemento Despesa':
			resultado.append(df.loc[int(rubrica[4:6])][0])
	return resultado

def gera_estatisticas(anomalias,grupos):
	df = DataFrame(columns=['municipio','grupo','qtd_rubricas','qtd_meses'])
	for municipio in tqdm(anomalias['municipio'].unique()):
		if len(df.index) > 0:
			df.loc[df.index[-1]+1] = [municipio, acha_grupo_do_municipio(municipio,grupos), len(anomalias[anomalias['municipio'] == municipio]['rubrica'].unique()), len(anomalias[anomalias['municipio'] == municipio]['mes'].unique()) ]
		else:
			df.loc[0] = [municipio, acha_grupo_do_municipio(municipio,grupos), len(anomalias[anomalias['municipio'] == municipio]['rubrica'].unique()), len(anomalias[anomalias['municipio'] == municipio]['mes'].unique()) ]
	return df