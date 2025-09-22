# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# SUPERVISED MACHINE LEARNING: MODELOS LOGÍSTICOS BINÁRIOS E MULTINOMIAIS
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8

# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')

## random forest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Carregar o dataset com todos os municípios do Brasil
# -------------------------------------------------------------------
# Ajuste o separador ('sep') conforme o formato do seu CSV (',' ou ';')
df = pd.read_csv('C:\MBA\data-set-populacao_indice-normalizado-13-04-2025.csv',delimiter=',')

# -------------------------------------------------------------------
# 2. Identificar colunas principais
# -------------------------------------------------------------------
# Alvo (y)
target_col = 'CONTAMINACAO_INDICE'
# Colunas de identificação
state_col = 'UF'
muni_col = 'MUNICIPIO'

# -------------------------------------------------------------------
# 3. Preparar X (features) e y (target)
# -------------------------------------------------------------------
# Extrai y como inteiro (0 ou 1)
y = df[target_col].astype(int)

# Seleciona todas as colunas numéricas, removendo alvo e identificadores
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col, 'DENGUE'])

# renomear as colunas
"""
num_cols = len(df.columns)
num_cols
df.columns = [f'F{i+1}' for i in range(num_cols)]
df.describe()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))"""
# -------------------------------------------------------------------
# 4. Dividir em treino e teste (70% train, 30% test) com estratificação
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------------------
# 5. Configurar Random Forest e GridSearchCV para otimização de hiperparâmetros
# -------------------------------------------------------------------
# Instancia o classificador
rf = RandomForestClassifier(random_state=42)

# Grade de hiperparâmetros a testar
param_grid = {
    'n_estimators': [100, 200],           # número de árvores
    'max_depth': [None, 5, 10],           # profundidade máxima das árvores
    'min_samples_split': [2, 5],          # min amostras para dividir um nó
    'min_samples_leaf': [1, 2]            # min amostras em uma folha
}

# GridSearch com validação cruzada (cv=5), otimizando pela AUC
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Ajusta o GridSearch aos dados de treino
grid_search.fit(X_train, y_train)

# -------------------------------------------------------------------
# 6. Extrair o melhor modelo e hiperparâmetros
# -------------------------------------------------------------------
best_rf = grid_search.best_estimator_
print("Melhores hiperparâmetros:", grid_search.best_params_)

# -------------------------------------------------------------------
# 7. Avaliar desempenho no conjunto de teste
# -------------------------------------------------------------------
# Previsões de classe
y_pred = best_rf.predict(X_test)
# Previsões de probabilidade para curva ROC
y_proba = best_rf.predict_proba(X_test)[:, 1]

# Relatório de classificação
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------------------------
# 8. Plotar Curva ROC e calcular AUC
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Supondo que você já treinou seu modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Importância das variáveis
importancias = modelo.feature_importances_
variaveis = X_train.columns

# Plotar
importancia_df = pd.DataFrame({'Variavel': variaveis, 'Importancia': importancias})
importancia_df = importancia_df.sort_values('Importancia', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importancia_df['Variavel'], importancia_df['Importancia'])
plt.xlabel("Importância")
plt.title("Importância das Variáveis - Random Forest")
plt.gca().invert_yaxis()
plt.show()

tabela_importancias = pd.DataFrame({
    'Variavel': variaveis,
    'Importancia': importancias
})

# Ordenando da variável mais importante para a menos importante
tabela_importancias = tabela_importancias.sort_values(by='Importancia', ascending=False)

# Exibindo a tabela
print(tabela_importancias)

# -------------------------------------------------------------------
# 9. Identificar e exibir as top 10 features mais importantes
# -------------------------------------------------------------------
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\nTop 10 Feature Importances:\n", importances.head(15))

# -------------------------------------------------------------------
# 10. Calcular e listar probabilidade de contaminação para cada município
# -------------------------------------------------------------------
df['PROB_CONTAMINACAO'] = best_rf.predict_proba(X)[:, 1]

top100 = df[[state_col, muni_col, 'PROB_CONTAMINACAO']].sort_values(
    by='PROB_CONTAMINACAO', ascending=False
).head(50)

print("\nTop 10 municípios por probabilidade de contaminação:\n", top100)

top100 = df[[state_col, muni_col, 'PROB_CONTAMINACAO']].sort_values(
    by='PROB_CONTAMINACAO', ascending=True
).head(10)

print("\nTop 10 municípios por probabilidade de contaminação:\n", top100)
top100.to_csv("top_6000.csv", sep=",")
top100.describe()

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Plotar histograma para cada variável numérica
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribuição da variável: {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()


# Calcular a matriz de correlação
correlation_matrix = X.corr().abs()

# Selecionar apenas a parte superior da matriz de correlação (sem duplicatas)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Encontrar colunas com correlação maior que 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Remover colunas com alta correlação
X_reduced = X.drop(columns=to_drop)

# Confirmar colunas removidas e shape final
to_drop, X.shape, X_reduced.shape

correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[PARTE CONCEITUAL]:
#############################################################################
#                             CURVA SIGMOIDE                                #
#############################################################################

# Estabelecendo uma função para a probabilidade de ocorrência de um evento

from math import exp

# Estabelecendo uma função para a probabilidade de ocorrência de um evento
def prob(z):
    return 1 / (1 + exp(-z))

# In[SIGMOIDE]: Plotando a curva sigmoide teórica de ocorrência de um evento
#para um range do logito z entre -5 e +5

logitos = []
probs = []

for i in np.arange(-5, 6):
    logitos.append(i)
    probs.append(prob(i))
    
df = pd.DataFrame({'logito': logitos, 'probs': probs})

# Interpolação spline (smooth probability line)
spline = UnivariateSpline(df['logito'], df['probs'], s=0)

logitos_smooth = np.linspace(df['logito'].min(), df['logito'].max(), 500)
probs_smooth = spline(logitos_smooth)

plt.figure(figsize=(15,10))
plt.plot(logitos_smooth, probs_smooth, color='royalblue',
         linestyle='--', label='Prob. Evento')
plt.scatter(df['logito'], df['probs'], color='royalblue', marker='o', s=250)
plt.axhline(y = df.probs.mean(), color = 'grey', linestyle = ':', xmax = .5)
plt.axvline(x = 0, color = 'grey', linestyle=':', ymax = 0.5)
plt.xlabel("Logito Z", fontsize=20)
plt.ylabel("Probabilidade", fontsize=20)
plt.xticks(np.arange(-5, 6), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.legend(fontsize=18, loc='center right')
plt.show()


# In[EXEMPLO 1]:
#############################################################################
#                      REGRESSÃO LOGÍSTICA BINÁRIA                          #                  
#               EXEMPLO 1 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_dengue = pd.read_csv('C:\MBA\data-set-populacao_indice-normalizado-13-04-2025-goias.csv',delimiter=',')
df_dengue.dtypes

# converte para float
"""
colunas_ajuste = df_dengue.columns
colunas_ajuste

for col in colunas_ajuste:
    if col != "MUNICIPIO" and col != "UF":
        # df_dengue[col] = df_dengue[col].str.replace(',', '.').str.strip()
        df_dengue[col] = pd.to_numeric(df_dengue[col], errors='coerce')
"""

# Características das variáveis do dataset
df_dengue.info()

# Estatísticas univariadas
df_dengue.describe()

# In[1.1]: Tabela de frequências absolutas da variável 'atrasado'

df_dengue['CONTAMINACAO_INDICE'].value_counts().sort_index()

# verificar se ha valores nulos
df_dengue.isnull().sum()
df_dengue[df_dengue.columns[df_dengue.isnull().any()]]
df_dengue[df_dengue.isnull().any(axis=1)]

# In[1.2]: Estimação de um modelo logístico binário pela função 'smf.glm'
#('statsmodels.formula.api')
"""
formula_bk = 

CONTAMINACAO_INDICE ~ 
TOTAL +
COM_RESPONSAVEL_OU_CONJUGE_ANALFABETO +
COM_SANEAMENTO_INADEQUADO +
COM_RESPONSAVEL_OU_CONJUGE_ANALFABETO_E_SANEAMENTO_INADEQUADO +
ADEQUADO +
SEMI_ADEQUADO +
INADEQUADO +
ATE_RS70 +
ATE_SM_RS128 +
ATE_SM_RS255 +
ATE_60PC_DA_MEDIANA_RS225 +
POPULACAO_RESIDENTE_EM_DOMICILIOS_PARTICULARES_PERMANENTES +
ATE_70_RS +
ATE_SALARIO_MINIMO_127_50_RS +
ATE_SALARIO_MINIMO_255_00_RS +
ATE_60PC_DA_MEDIANA_0_BRASIL_TOTAL_225_00_RS +
BRANCA_PRETA_A_B +
BRANCA_PARDA_A_C +
BRANCA_AMARELA_A_D +
BRANCA_INDIGENA_A_E +
PRETA_PARDA_B_C +
HOMEM_A_VALOR_MEDIO +
MULHER_B_VALOR_MEDIO +
RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIO_A_B +
RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIANO_C_D +
DENGUE

"""

formula = """
CONTAMINACAO_INDICE ~
TOTAL
"""

modelo_dengue_binomial = smf.glm(formula=formula, data=df_dengue, family=sm.families.Binomial()).fit()

# Parâmetros do 'modelo_atrasos'
modelo_dengue_binomial.summary()
df_dengue = df_dengue.drop(columns=[
    "MUNICIPIO", "UF"
  ]
)


correlation_matrix = df_dengue.iloc[:,1:4].corr()
correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Adimplente", "Inadimplente"],
            yticklabels=["Adimplente", "Inadimplente"])
plt.title(f"Matriz de Confusão - {classifier.class.name}")
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.show()

import pingouin as pg

correlation_matrix2 = pg.rcorr(df_dengue, method='pearson',
                               upper='pval', decimals=6,
                               pval_stars={0.01: '***',
                                           0.05: '**',
                                           0.10: '*'})
correlation_matrix2

import seaborn as sns

# renomeia colunas
num_cols = len(df_dengue.columns)
num_cols
df_dengue.columns = [f'F{i+1}' for i in range(num_cols)]
df_dengue.describe()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

sub_sample_corr = df_dengue.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('Matriz de correlação com dados balanceados', fontsize=14)
plt.show()



# In[1.3]: Outputs do modelo pela função 'summary_col'

summary_col([modelo_dengue_binomial],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

# In[1.4]: Fazendo predições para o 'modelo_atrasos'.
#Exemplo: qual a probabilidade média de se chegar atrasado quando o
#trajeto tem 7 km e passa-se por 10 semáforos no percurso?

modelo_dengue_binomial.predict(pd.DataFrame({'TOTAL':[10000]}))

# In[1.5]: Construção de uma matriz de confusão

# Adicionando os valores previstos de probabilidade na base de dados
df_dengue['phat'] = modelo_dengue_binomial.predict()

# Visualização da base de dados com a variável 'phat'
df_dengue

# In[1.6]: Gráficos com ajustes entre a variável dependente e a variável 'sem'
    
# Ajuste linear entre a variável dependente e a variável 'sem' (Gráfico errado:
#apenas para fins didáticos)
"""
plt.figure(figsize=(15,10))
sns.regplot(x=df_dengue['DENGUE'], y=df_dengue['CONTAMINACAO_INDICE'],
            ci=None, marker='o',
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('DENGUE', fontsize=20)
plt.ylabel('CONTAMINACAO_INDICE', fontsize=20)
plt.xticks(np.arange(0, df_dengue['DENGUE'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()"""

# In[1.7]: Ajuste logístico determinístico entre a variável dependente e a
#variável 'sem'

# Sigmoide

plt.figure(figsize=(15,10))
sns.regplot(x=df_dengue['CONTAMINACAO_INDICE'], y=df_dengue['phat'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('CONTAMINACAO_INDICE', fontsize=20)
plt.ylabel('phat', fontsize=20)
plt.xticks(np.arange(0, df_dengue['phat'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

# In[1.8]: Ajuste logístico probabilístico entre a variável dependente e a
#variável 'sem'

# Sigmoide

plt.figure(figsize=(15,10))
sns.regplot(x=df_dengue['CONTAMINACAO_INDICE'], y=df_dengue['phat'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('CONTAMINACAO_INDICE', fontsize=20)
plt.ylabel('phat', fontsize=20)
plt.xticks(np.arange(0, df_dengue['phat'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show

# In[1.9]: Construção de função para a definição da matriz de confusão

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

# In[1.10]: Matrizes de confusão propriamente ditas

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_dengue['CONTAMINACAO_INDICE'],
                predicts=df_dengue['phat'], 
                cutoff=0.5)

# Matriz de confusão para cutoff = 0.3
matriz_confusao(observado=df_dengue['CONTAMINACAO_INDICE'],
                predicts=df_dengue['phat'], 
                cutoff=0.4)

# Matriz de confusão para cutoff = 0.7
matriz_confusao(observado=df_dengue['CONTAMINACAO_INDICE'],
                predicts=df_dengue['phat'], 
                cutoff=0.7)

# In[1.11]: Igualando critérios de especificidade e de sensitividade

# Tentaremos estabelecer um critério que iguale a probabilidade de
#acerto daqueles que chegarão atrasados (sensitividade) e a probabilidade de
#acerto daqueles que não chegarão atrasados (especificidade).

# ATENÇÃO: o que será feito a seguir possui fins didáticos, apenas. DE NENHUMA
#FORMA, o procedimento garante a maximização da acurácia do modelo!

# Criação da função 'espec_sens' para a construção de um dataset com diferentes
#valores de cutoff, sensitividade e especificidade:

def espec_sens(observado,predicts):
    
    # Adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # Range dos cutoffs a serem analisados em steps de 0.01
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

# In[1.12]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados (dataframe 'dados_plotagem')

dados_plotagem = espec_sens(observado = df_dengue['CONTAMINACAO_INDICE'],
                            predicts = df_dengue['phat'])
dados_plotagem

# In[1.13]: Plotagem de um gráfico que mostra a variação da especificidade e da
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

# In[1.14]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_dengue['CONTAMINACAO_INDICE'], df_dengue['phat'])
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


# In[EXEMPLO 2]:
#############################################################################
#           REGRESSÃO LOGÍSTICA BINÁRIA E PROCEDIMENTO STEPWISE             #        
#                EXEMPLO 2 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_challenger = pd.read_csv('challenger.csv',delimiter=',')
df_challenger

# Características das variáveis do dataset
df_challenger.info()

# Estatísticas univariadas
df_challenger.describe()

# desgaste: quantidade de vezes em que ocorreu stress térmico
# temperatura: temperatura de lançamento (graus ºF)
# pressão: pressão de verificação de vazamento (psi: libra-força por
    #polegada ao quadrado)
# t: teste para o lançamento (id)

# In[2.1]: Criação da variável dependente binária 'falha'

# Não há uma variável binária para servir como uma variável dependente, certo?
# Então vamos criá-la, considerando a ocorrência de desgastes de peças como a
#ocorrência de um evento que chamaremos de 'falha':

df_challenger.loc[df_challenger['desgaste'] != 0 , 'falha'] = 1
df_challenger.loc[df_challenger['desgaste'] == 0, 'falha'] = 0

# Transformando a variável 'falha' para o tipo 'int' (poderia também deixar
#como 'float'), a fim de que seja possível estimar o modelo por meio da
#função 'sm.Logit.from_formula'

df_challenger.info()

df_challenger['falha'] = df_challenger['falha'].astype('int64')

df_challenger.info()

df_challenger

# In[2.2]: Gráfico 'pairplot' com scatters e KDEs por 'falha'

cores_desejadas = {0: 'springgreen', 1: 'magenta'}

g = sns.pairplot(df_dengue[['CONTAMINACAO_INDICE','TOTAL']], hue='CONTAMINACAO_INDICE',
                 palette=cores_desejadas)
g.fig.set_size_inches(8, 6)
plt.show()


# In[2.3]: Estimação do modelo logístico binário pela função 'sm.Logit.from_formula'
#('statsmodels.api')

# O modelo a seguir também pode ser estimado por meio da função 'smf.glm'
#('statsmodels.formula.api')

modelo_dengue_stepwise = sm.Logit.from_formula(formula, df_dengue).fit()

# Parâmetros do 'modelo_challenger'
modelo_dengue_stepwise.summary()

# In[2.4]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
step_dengue = stepwise(modelo_dengue_stepwise, pvalue_limit=0.05)

# In[2.5]: Fazendo predições para o modelo 'step_challenger'

# Exemplo 1: qual a probabilidade média de falha a 70ºF (~21.11ºC)?
step_dengue.predict(pd.DataFrame({'TOTAL':[100000]}))


# In[2.6]: Atribuindo uma coluna no dataframe para os resultados

df_dengue['phat'] = step_dengue.predict()

df_dengue

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


# In[3.12]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_dengue['CONTAMINACAO_INDICE'],
                                df_dengue['phat'])
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

# In[1.3]: Outputs do modelo pela função 'summary_col'

summary_col([step_dengue],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

from sklearn.metrics import roc_curve, auc

y_scores = step_dengue.fittedvalues

# E o valor real do desfecho
y_true = df_dengue["CONTAMINACAO_INDICE"].astype(int)

# Calcula FPR (False Positive Rate), TPR (True Positive Rate) e thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calcula a área sob a curva (AUC)
roc_auc = auc(fpr, tpr)

# Plota a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # linha de referência
plt.xlabel('Taxa de falsos positivos (FPR)')
plt.ylabel('Taxa de verdadeiros positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[EXEMPLO 4]:
#############################################################################
#                      REGRESSÃO LOGÍSTICA MULTINOMIAL                      #
#                 EXEMPLO 4 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_atrasado_multinomial = pd.read_csv('atrasado_multinomial.csv',delimiter=',')
df_atrasado_multinomial

# Características das variáveis do dataset
df_atrasado_multinomial.info()

# Estatísticas univariadas
df_atrasado_multinomial.describe()

# In[4.1]: Note que a variável Y 'atrasado' está definida como objeto

# Tabela de frequências absolutas da variável 'atrasado' com labels
df_atrasado_multinomial['atrasado'].value_counts().sort_index()

# Criando uma variável 'atrasado2' a partir da variável 'atrasado',
#com labels iguais a 0, 1 e 2 e com tipo 'int' (poderia também ser do tipo
#'float'), a fim de que seja possível estimar o modelo por meio
#da função 'MNLogit' do pacote 'statsmodels.discrete.discrete_model'

df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'nao chegou atrasado',
                            'atrasado2'] = 0 #categoria de referência

df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'chegou atrasado primeira aula',
                            'atrasado2'] = 1

df_atrasado_multinomial.loc[df_atrasado_multinomial['atrasado']==
                            'chegou atrasado segunda aula',
                            'atrasado2'] = 2

# Definição do tipo 'int' para a variável dependente 'atrasado2'
df_atrasado_multinomial['atrasado2'] =\
    df_atrasado_multinomial['atrasado2'].astype('int64')

df_atrasado_multinomial.info()

df_atrasado_multinomial

# In[4.2]: Estimação do modelo logístico multinomial

x = df_atrasado_multinomial.drop(columns=['estudante','atrasado','atrasado2'])
y = df_atrasado_multinomial['atrasado2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_atrasado = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo 'modelo_atrasado'
modelo_atrasado.summary()

# In[4.3]: Vamos definir uma função 'Qui2' para se extrair a estatística geral
# do modelo

def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,1)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df

# In[4.4]: Estatística geral do 'modelo_atrasado'

Qui2(modelo_atrasado)

# In[4.5]: Fazendo predições para o 'modelo_atrasado'

# Exemplo: qual a probabilidade média de atraso para cada categoria da
#variável dependente, se o indivíduo tiver que percorrer 22km e passar
#por 12 semáforos?

# No nosso exemplo, temos os seguintes labels para a variável 'atrasado2':
# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

resultado = modelo_atrasado.predict(pd.DataFrame({'const':[1],
                                                   'dist':[22],
                                                   'sem':[12]})).round(4)

resultado

# Uma maneira de identificar a classe do resultado de acordo com o 'predict'
resultado.idxmax(axis=1)

# In[4.6]: Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_atrasado.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_atrasado_multinomial = pd.concat([df_atrasado_multinomial, phats], axis=1)
df_atrasado_multinomial

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_atrasado_multinomial['predicao'] = predicao
df_atrasado_multinomial

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==0,
                            'predicao_label'] ='não chegou atrasado'
df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==1,
                            'predicao_label'] ='chegou atrasado primeira aula'
df_atrasado_multinomial.loc[df_atrasado_multinomial['predicao']==2,
                            'predicao_label'] ='chegou atrasado segunda aula'

df_atrasado_multinomial

# In[4.7]: Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_atrasado_multinomial,
                       index=['predicao_label'],
                       columns=['atrasado'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table

# In[4.8]: Visualização, para fins didáticos, do objeto 'table' (dataframe)
#no ambiente Plots

from tabulate import tabulate
tabela = tabulate(table, headers='keys', tablefmt='grid', numalign='center')

plt.figure(figsize=(8, 3))
plt.text(0.1, 0.1, tabela, {'family': 'monospace', 'size': 15})
plt.axis('off')
plt.show()

# In[4.9]: Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia = table.diagonal().sum()/table.sum()
acuracia

# In[4.10]: Plotagens das probabilidades

# Plotagem das smooth probability lines para a variável 'dist'

# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

plt.figure(figsize=(15,10))

# Plot para "não chegou atrasado"
sns.regplot(x='dist', y=df_atrasado_multinomial[0],
            data=df_atrasado_multinomial, ci=False, order=4,
            line_kws={'color':'indigo', 'linewidth':4,
                      'label':'não chegou atrasado'},
            scatter_kws={'color':'indigo', 's':80, 'alpha':0.5})

# Plot para "chegou atrasado na primeira aula"
sns.regplot(x='dist', y=df_atrasado_multinomial[1],
            data=df_atrasado_multinomial, ci=None, order=4,
            line_kws={'color':'darkgreen', 'linewidth':4,
                      'label':'chegou atrasado na primeira aula'},
            scatter_kws={'color':'darkgreen', 's':80, 'alpha':0.5})

# Plot para "chegou atrasado na segunda aula"
sns.regplot(x='dist', y=df_atrasado_multinomial[2],
            data=df_atrasado_multinomial, ci=None, order=4,
            line_kws={'color':'darkorange', 'linewidth':4,
                      'label':'chegou atrasado na segunda aula'},
            scatter_kws={'color':'darkorange', 's':80, 'alpha':0.5})

plt.xlabel('Distância Percorrida', fontsize=18)
plt.ylabel('Probabilidades', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='center left', fontsize=14)
plt.show()

# In[4.11]: Plotagens das probabilidades

# Plotagem das smooth probability lines para a variável 'sem'

# 0: não chegou atrasado
# 1: chegou atrasado primeira aula
# 2: chegou atrasado segunda aula

plt.figure(figsize=(15,10))

# Plot para "não chegou atrasado"
sns.regplot(x='sem', y=df_atrasado_multinomial[0],
            data=df_atrasado_multinomial, ci=None, order=4,
            line_kws={'color':'indigo', 'linewidth':4,
                      'label':'não chegou atrasado'},
            scatter_kws={'color':'indigo', 's':80, 'alpha':0.5})

# Plot para "chegou atrasado na primeira aula"
sns.regplot(x='sem', y=df_atrasado_multinomial[1],
            data=df_atrasado_multinomial, ci=None, order=4,
            line_kws={'color':'darkgreen', 'linewidth':4,
                      'label':'chegou atrasado na primeira aula'},
            scatter_kws={'color':'darkgreen', 's':80, 'alpha':0.5})

# Plot para "chegou atrasado na segunda aula"
sns.regplot(x='sem', y=df_atrasado_multinomial[2],
            data=df_atrasado_multinomial, ci=None, order=4,
            line_kws={'color':'darkorange', 'linewidth':4,
                      'label':'chegou atrasado na segunda aula'},
            scatter_kws={'color':'darkorange', 's':80, 'alpha':0.5})

plt.xlabel('Quantidade de Semáforos', fontsize=18)
plt.ylabel('Probabilidades', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper center', fontsize=14)
plt.show()

# In[4.12]: Plotagem tridimensional para cada probabilidade de ocorrência de
#cada categoria da variável dependente

# Probabilidades de não se chegar atrasado (função 'go' do pacote 'plotly')

import plotly.io as pio
pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[0],
    opacity=1, intensity=df_atrasado_multinomial[0],
    colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='não chegou atrasado'))

plot_figure.show()

# In[4.13]: Plotagem tridimensional para cada probabilidade de ocorrência de
#cada categoria da variável dependente

# Probabilidades de se chegar atrasado à primeira aula (função 'go' do pacote
#'plotly')

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[1],
    opacity=1, intensity=df_atrasado_multinomial[1],
    colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='chegou atrasado à primeira aula'))

plot_figure.show()

# In[4.14]: Plotagem tridimensional para cada probabilidade de ocorrência de
#cada categoria da variável dependente

# Probabilidades de se chegar atrasado à segunda aula (função 'go' do pacote
#'plotly')

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[2],
    opacity=1, intensity=df_atrasado_multinomial[2],
    colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='dist',
                        yaxis_title='sem',
                        zaxis_title='chegou atrasado à segunda aula'))

plot_figure.show()

# In[4.15]: Visualização das sigmoides tridimensionais em um único gráfico

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_atrasado_multinomial['dist'], 
    y=df_atrasado_multinomial['sem'],
    z=df_atrasado_multinomial[0],
    opacity=1,
    color='indigo')

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

trace_1 = go.Mesh3d(
            x=df_atrasado_multinomial['dist'], 
            y=df_atrasado_multinomial['sem'],
            z=df_atrasado_multinomial[1],
            opacity=1,
            color='darkgreen')

plot_figure.add_trace(trace_1)

trace_2 = go.Mesh3d(
            x=df_atrasado_multinomial['dist'], 
            y=df_atrasado_multinomial['sem'],
            z=df_atrasado_multinomial[2],
            opacity=1,
            color='darkorange')


plot_figure.add_trace(trace_2)

plot_figure.update_layout(
    template='plotly_dark',
    scene = dict(
        xaxis_title='dist',
        yaxis_title='sem',
        zaxis_title='probabilidades')
    )

plot_figure.show()

################################## FIM ######################################