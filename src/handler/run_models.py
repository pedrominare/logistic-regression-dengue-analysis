# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests
!pip install tensorflow
# Imported Libraries

from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada

from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('B:\\Documentos\\MBA\\data-set-populacao-brasil-inteiro-indice-normalizado.csv', encoding='latin-1', sep=';')
df.drop(['UF', 'MUNICIPIO', 'MEDIA_PONDERADA_INDICE'], axis=1, inplace=True)
df.head()

print(df.describe())

# Good No Null Values!
df.isnull().sum().max()

print(df.columns)

# The classes are heavily skewed we need to solve this issue later.
print('NAO CONTAMINACAO', round(df['CONTAMINACAO_INDICE'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('CONTAMINACAO', round(df['CONTAMINACAO_INDICE'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

colors = ["#0101DF", "#DF0101"]
#scaler = StandardScaler()
#df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

sns.countplot(x='CONTAMINACAO_INDICE', data=df, palette=colors)
plt.title('CONTAMINACAO_INDICE Distributions \n (0: NAO CONTAMINADO || 1: CONTAMINADO)', fontsize=14)

# skewed features
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['DENGUE'].values
time_val = df['CONTAMINACAO_INDICE'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('DENGUE Distribution', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('CONTAMINACAO_INDICE Distribution', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

# matriz de correlacao
correlacao = df.corr()

# inicio dos indices para colunas
colunas_originais = df.columns  # Guardar os nomes originais
indices_numericos = [f"F{i+1}" for i in range(len(colunas_originais))]  # Criar nomes F1, F2, ...
correlacao.index = indices_numericos
correlacao.columns = indices_numericos
plt.figure(figsize=(20, 10))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(ticks=range(len(indices_numericos)), labels=indices_numericos, rotation=45)
plt.yticks(ticks=range(len(indices_numericos)), labels=indices_numericos, rotation=0)
plt.title("Matriz de Correlação com Índices Numéricos")
plt.show()

referencia = dict(zip(indices_numericos, colunas_originais))
print("Referência dos Índices:")
for k, v in referencia.items():
    print(f"{k} → {v}")

print(df.describe())
print(df['TOTAL'])

indice_coluna = 0
for coluna in df.columns:
    if coluna != 'CONTAMINACAO_INDICE':
        plt.figure(figsize=(15, 8))
        sns.boxplot(x="CONTAMINACAO_INDICE", y=coluna, data=df, palette=colors)
        plt.title(f'CONTAMINACAO_INDICE vs {coluna}')
        plt.show()
    
    indice_coluna += 1
############    
# In[2.2]: Gráfico 'pairplot' com scatters e KDEs por 'CONTAMINACAO_INDICE'
plt.figure(figsize=(15,10))
cores_desejadas = {0: 'springgreen', 1: 'magenta'}
df['CONTAMINACAO_INDICE'].astype('int64')

for coluna in df.columns:
    print(coluna)
    if coluna != 'CONTAMINACAO_INDICE':
        g1 = sns.pairplot(df[['CONTAMINACAO_INDICE', coluna]], hue='CONTAMINACAO_INDICE', palette=cores_desejadas)
        g1.fig.set_size_inches(8, 6)
        plt.show()

print(df['TOTAL'])
modelo_df_dengue = sm.Logit.from_formula('CONTAMINACAO_INDICE ~ DENGUE / TOTAL', df).fit()

# Parâmetros do 'modelo_challenger'
modelo_df_dengue.summary()

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
step_dengue = stepwise(modelo_df_dengue, pvalue_limit=0.05)

# Exemplo 1: qual a probabilidade média de CONTAMINACAO_INDICE a 70ºF (~21.11ºC)?
step_dengue.predict(pd.DataFrame({'TOTAL':[100000]}))

####
# In[1.2]: Estimação de um modelo logístico binário pela função 'smf.glm'
#('statsmodels.formula.api')
print(df.columns)
modelo_atrasos = smf.glm(formula='CONTAMINACAO_INDICE ~ DENGUE / TOTAL', data=df, family=sm.families.Binomial()).fit()

# Parâmetros do 'modelo_atrasos'
modelo_atrasos.summary()
summary_col([modelo_atrasos],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

plt.figure(figsize=(15,10))
sns.regplot(x=df['DENGUE'], y=df['CONTAMINACAO_INDICE'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Quantidade de contaminados por dengue', fontsize=20)
plt.ylabel('CONTAMINACAO_INDICE', fontsize=20)
plt.xticks(np.arange(0, df['DENGUE'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show

# In[1.4]: Fazendo predições para o 'modelo_atrasos'.
#Exemplo: qual a probabilidade média de se chegar atrasado quando o
#trajeto tem 7 km e passa-se por 10 semáforos no percurso?

# qual a probabilidade de um individuo contrair dengue numa cidade com a populacao total de 10 habitantes
# e um total de 7 contaminados no municipio?
modelo_atrasos.predict(pd.DataFrame({'TOTAL':[1000], 'DENGUE':[10]}))
df['phat'] = modelo_atrasos.predict()

plt.figure(figsize=(15,10))
sns.regplot(x=df['DENGUE'], y=df['phat'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('DENGUE', fontsize=20)
plt.ylabel('phat', fontsize=20)
plt.xticks(np.arange(0, df['DENGUE'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show

plt.figure(figsize=(15,10))
sns.regplot(x=df['CONTAMINACAO_INDICE'], y=df['phat'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('CONTAMINACAO_INDICE', fontsize=20)
plt.ylabel('phat', fontsize=20)
plt.xticks(np.arange(0, df['CONTAMINACAO_INDICE'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df['CONTAMINACAO_INDICE'],
                predicts=df['phat'], 
                cutoff=0.5)

###### curva ROC ###
import pickle
import pandas as pd
from funcoes_ajuda import descritiva, relatorio_missing, \
    diagnóstico, avalia_clf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, \
    StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

print(df)
#%% Definindo a lista de features
variaveis = list(df.columns)
vResp = 'CONTAMINACAO_INDICE'

print(variaveis)
print(vResp)

#%% A análise descritiva é sempre um passo muito importante
'''
for var in variaveis:
    descritiva(df, var, vResp, 6)
'''

#%% Dividir a base em treino e teste
variaveis.remove(vResp)
variaveis.remove('phat')

#%% Refazendo as bases de treino e teste
X = df[variaveis]
y=df[vResp]
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1729)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Treinando a primeira árvore (correta agora)
arvore1 = DecisionTreeClassifier()
arvore1.fit(X_train, y_train)

#%% Avaliando a primeira árvore (correta)
avalia_clf(arvore1, y_train, X_train, rótulos_y=['NÃO CONTAMINADO', 'CONTAMINADO'],base='treino')
avalia_clf(arvore1, y_test, X_test, rótulos_y=['NÃO CONTAMINADO', 'CONTAMINADO'],base='teste')

#%%
path = arvore1.cost_complexity_pruning_path(X_train, y_train)  # CCP Path na base de treino
ccp_alphas, impurities = path.ccp_alphas, path.impurities

#%% Verificar se há duplicações nos ccp_alphas
print(len(ccp_alphas))
len(pd.Series(ccp_alphas).unique())
print(df)

# NERY
# Inicializando o scaler
scaler = StandardScaler()

# Separando os dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.values
y_test = y_test.values

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Aplicar o SelectKBest para selecionar as k melhores features
k = 10 # Número de features a serem selecionadas. Ajuste conforme necessário.
selector = SelectKBest(score_func=f_classif, k=k)  # Usando f_classif como função de pontuação (ANOVA)
X_selected = selector.fit_transform(X, y)

# Obter os nomes das colunas selecionadas
selected_features = X.columns[selector.get_support()]
print(f"Colunas selecionadas pelo SelectKBest (k={k}):\n{selected_features}")

# Criar DataFrames com as colunas selecionadas
X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
print(X_selected.head())

from sklearn.feature_selection import SelectKBest, f_classif

model = KNeighborsClassifier(n_neighbors=3)  # Usando 3 vizinhos
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Faz previsões
accuracy = model.score(X_test, y_test)  # Calcula a acurácia
print(f"Acurácia do modelo: {accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva para a curva ROC

# matriz de confusao
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["NÃO CONTAMINADO", "CONTAMINADO"],
            yticklabels=["NÃO CONTAMINADO", "CONTAMINADO"])
plt.title(f"Matriz de Confusão {classifier.__class__.__name__}")
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.show()

########################################
# In[3.7]: Construção de função para a definição da matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# In[3.8]: Construção da matriz de confusão

# Adicionando os valores previstos de probabilidade na base de dados
df['phat'] = step_modelo_fidelidade.predict()

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df['CONTAMINACAO_INDICE'],
                predicts=df['phat'],
                cutoff=0.50)

# In[3.9]: Igualando critérios de especificidade e de sensitividade

# Tentaremos estabelecer um critério que iguale a probabilidade de
#acerto daqueles que chegarão atrasados (sensitividade) e a probabilidade de
#acerto daqueles que não chegarão atrasados (especificidade).

# ATENÇÃO: o que será feito a seguir possui fins didáticos, apenas. DE NENHUMA
#FORMA o procedimento garante a maximização da acurácia do modelo!

# Criação da função 'espec_sens' para a construção de um dataset com diferentes
#valores de cutoff, sensitividade e especificidade:

def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

# In[3.10]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados

dados_plotagem = espec_sens(observado = df['CONTAMINACAO_INDICE'],
                            predicts = df['phat'])
dados_plotagem

# In[3.11]: Plotagem de um gráfico que mostra a variação da especificidade e da
#sensitividade em função do cutoff

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()

# In[3.12]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df['CONTAMINACAO_INDICE'],
                                df['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()












##################################################################




# New_df is from the random undersample data (fewer instances)
X = df.drop('CONTAMINACAO_INDICE', axis=1)
y = df['CONTAMINACAO_INDICE']


# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()

####

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()

#################################################################################
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['TOTAL'].values
time_val = df['DENGUE'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('DISTRIBUICAO DO TOTAL DA POPULACAO', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('DISTRIBUICAO DE CONTAMINACAO DE DENGUE DA POPULACAO', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['TOTAL_SCALED'] = rob_scaler.fit_transform(df['TOTAL'].values.reshape(-1,1))
df['DENGUE_SCALED'] = rob_scaler.fit_transform(df['DENGUE'].values.reshape(-1,1))

df.drop(['DENGUE','TOTAL'], axis=1, inplace=True)

scaled_amount = df['TOTAL_SCALED']
scaled_time = df['DENGUE_SCALED']

df.drop(['DENGUE_SCALED', 'TOTAL_SCALED'], axis=1, inplace=True)
df.insert(0, 'TOTAL_SCALED', scaled_amount)
df.insert(1, 'DENGUE_SCALED', scaled_time)

# Amount and Time are Scaled!

df.head()

print('NAO CONTAMINACAO', round(df['CONTAMINACAO_INDICE'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('CONTAMINACAO', round(df['CONTAMINACAO_INDICE'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('CONTAMINACAO_INDICE', axis=1)
y = df['CONTAMINACAO_INDICE']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# Check the Distribution of the labels

# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.
# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['CONTAMINACAO_INDICE'] == 1]
non_fraud_df = df.loc[df['CONTAMINACAO_INDICE'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

print('Distribution of the CONTAMINACAO_INDICE in the subsample dataset')
print(new_df['CONTAMINACAO_INDICE'].value_counts()/len(new_df))

sns.countplot('CONTAMINACAO_INDICE', data=new_df, palette=colors)
plt.title('Equally Distributed CONTAMINACAO_INDICE', fontsize=14)
plt.show()

# Make sure we use the subsample in our correlation

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()

f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="CONTAMINACAO_INDICE", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="CONTAMINACAO_INDICE", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="CONTAMINACAO_INDICE", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="CONTAMINACAO_INDICE", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()
