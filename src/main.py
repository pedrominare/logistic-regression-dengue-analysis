# monta os DFs
import pandas as pd
import logging
import os

from data.web_scraping import WebScraping
from handler.df_handler import DfHandler, agrupa_df_com_populacao_total, cria_coluna_contaminacao_indice, \
    normalizar_colunas
from handler.logistic_regression_handler import LogisticRegressionHandler
from utils.df_helper import normalizar_string
from data.municipios import municipios_goias

from memory_monitor.memory_monitor import MemoryMonitor

# Configuração de logging
logging.basicConfig(level=logging.INFO)


def main():
    obj_memory_usage = MemoryMonitor()
    obj_memory_usage.add_checkpoint(name="Codigo iniciado!")

    # arquivos XLSX com datasets
    xlsx_casos_dengue = os.path.join(os.getcwd(), "dengue-casos-notificados-all.xlsx")
    xlsx_municipios = os.path.join(os.getcwd(), "DATA-SET-2010.xlsx")
    ano_interesse = "2010"

    obj_dengue = DfHandler(
        xlsx_name=xlsx_casos_dengue, logging=logging, memory_usage=obj_memory_usage, ano=ano_interesse
    )
    obj_dengue.build_df(tipo="dengue")

    obj_municipios = DfHandler(
        xlsx_name=xlsx_municipios, logging=logging, memory_usage=obj_memory_usage, skiprows=2
    )

    obj_municipios.build_df(tipo="municipio")
    obj_municipios.cria_lista_df_municipios(municipios_goias)

    df_somatorio_dengue = obj_dengue.agrupa_casos_dengue(
        casos=["DENGUE", "DENGUE GRAVE"]
    )
    df_data_set = obj_municipios.agrupa_casos_dengue_com_municipios(df_somatorio_dengue)

    # obtem dados da populacao de cada municipio
    obj_web_scraping_municipios_goias = WebScraping()
    
    # o codigo do estado de goias eh 52
    df_populacao_municipios_goias = obj_web_scraping_municipios_goias.obtem_populacao_estado(cod_uf='52')
    df_populacao_municipios_goias.to_csv('data-set-completo-original.csv', index=False, encoding='latin-1', sep=';')

    # dropa colunas que nao serao utilizadas na analise
    df_populacao_municipios_goias.drop(
        columns=[
            'URBANA',
            'URBANANA SEDE MUNICIPAL',
            'RURAL',
            'TOTALPERCENTUAL',
            'URBANAPERCENTUAL',
            'URBANANA SEDE MUNICIPALPERCENTUAL',
            'RURALPERCENTUAL',
            'AREA TOTALKM2',
            'DENSIDADE DEMOGRAFICA DA UNIDADE TERRITORIALHAB/KM2'
        ],
        axis=1,
        inplace=True
    )
    
    df_data_set = agrupa_df_com_populacao_total(
        df_populacao=df_populacao_municipios_goias,
        df_dataset=df_data_set
    )

    df_data_set = cria_coluna_contaminacao_indice(df_data_set)

    # normalizar colunas, tirar acentos, etc
    df_data_set = normalizar_colunas(df_data_set)

    # escreve para CSV
    df_data_set.to_csv('data-set-populacao_indice-normalizado.csv', index=False, encoding='latin-1', sep=';')

    # caso o dataset ja tenha sido criado
    df_data_set = pd.read_csv('data-set-populacao_indice.csv', encoding='latin-1', sep=';')

    obj_logistic_regression = LogisticRegressionHandler(
        test_size=0.2, random_state=42
    )

    obj_logistic_regression.logistic_regression(
        df_dataset=df_data_set
    )

    """
    As colunas de 1 municipio agora dizem respeito a cada planilha, individualmente.
    A intenção é organizar colunas como variáveis, unicas para cada indicador socioeconomico.
    """

    print("pare aqui")


if __name__ == "__main__":
    main()

"""
# obtem nomes das planilhas do xlsx
sheet_names_casos_dengue = get_sheet_names(xlsx_casos_dengue)
sheet_names_municipios = get_sheet_names(xlsx_municipios)

lista_dfs_municipios = []
lista_dfs_casos_dengue = []

for sheet_municio in sheet_names_municipios:
    df_municipio = BuildDf(
        xlsx_name=xlsx_municipios,
        sheet_name=sheet_municio,
    )
    df_municipio.df["indice_socioeconomico"] = sheet_municio  # adiciona o nome da planilha em todas as observacoes
    df_municipio.get_variables()
    dados_municipio = obtem_dados_municipio(
        df=df_municipio.df,
        municipio=municipio_interesse,
        colunas=df_municipio.variables
    )
    lista_dfs_municipios.append(dados_municipio)
    del df_municipio

# unir todos os dataframes de municipios com suas respectivas colunas
df_concatenado = pd.concat(lista_dfs_municipios, ignore_index=True)

for sheet_municio in sheet_names_casos_dengue:
    df_dengue = BuildDf(
        xlsx_name=xlsx_casos_dengue,
        sheet_name=sheet_municio,
        skiprows=None
    )
    lista_dfs_casos_dengue.append(df_dengue)
    del df_dengue
"""
print("Fim")
