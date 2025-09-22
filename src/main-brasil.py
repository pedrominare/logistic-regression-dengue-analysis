# monta os DFs
import pandas as pd
import logging
import os

from data.web_scraping import WebScraping
from handler.df_handler import DfHandler, agrupa_df_com_populacao_total, cria_coluna_contaminacao_indice, \
    normalizar_colunas
from handler.logistic_regression_handler import LogisticRegressionHandler
from utils.df_helper import normalizar_string
from data.municipios import municipios_goias, municipios_brasil, municipios_brasil_com_ou_sem_dengue

from memory_monitor.memory_monitor import MemoryMonitor

# Configuração de logging
logging.basicConfig(level=logging.INFO)


def main():
    obj_memory_usage = MemoryMonitor()
    obj_memory_usage.add_checkpoint(name="Codigo iniciado!")


    # arquivos XLSX com datasets
    xlsx_casos_dengue = os.path.join(os.getcwd(), "brasil-inteiro", "DENGUE_BRASIL_CONTAMINADOS.xlsx")
    xlsx_municipios = os.path.join(os.getcwd(), "brasil-inteiro", "DATA-SET-2010_DADOS-SALVOS.xlsx")

    obj_dengue = DfHandler(
        xlsx_name=xlsx_casos_dengue, logging=logging, memory_usage=obj_memory_usage, ano="2010"
    )
    obj_dengue.build_df(tipo="dengue")

    obj_municipios = DfHandler(
        xlsx_name=xlsx_municipios, logging=logging, memory_usage=obj_memory_usage, skiprows=2
    )

    obj_municipios.build_df(tipo="municipio")
    obj_municipios.cria_lista_df_municipios_otimizado(municipios_brasil_com_ou_sem_dengue)

    obj_municipios = DfHandler(
        xlsx_name=xlsx_municipios, logging=logging, memory_usage=obj_memory_usage, skiprows=2
    )
    obj_municipios.municipios_concatenados = pd.read_csv('data-set-completo-municipios-brasil-todos-municipios.csv', encoding='latin-1', sep=';')

    df_somatorio_dengue = obj_dengue.agrupa_casos_dengue_brasil_inteiro()

    df_data_set = obj_municipios.agrupa_casos_dengue_com_municipios_brasil_todo(obj_dengue)

    # obtem dados da populacao de cada municipio
    obj_web_scraping_municipios_brasil = WebScraping()
    
    # o codigo do estado de goias eh 52
    df_populacao_municipios_brasil = obj_web_scraping_municipios_brasil.obtem_populacao_estado()
    df_populacao_municipios_brasil.to_csv('data-set-populacao-total-municipios-brasil.csv', index=False, encoding='latin-1', sep=';')

    df_data_set = pd.read_csv('data-set-pronto-dengue-municipios.csv', encoding='latin-1', sep=';')
    df_data_set.rename(columns={'CONTAMINADOS': 'DENGUE'}, inplace=True)
    df_populacao_municipios_brasil = pd.read_csv('data-set-populacao-total-municipios-brasil.csv', encoding='latin-1', sep=';')

    # dropa colunas que nao serao utilizadas na analise
    df_populacao_municipios_brasil.drop(
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

    # dropa colunas que nao serao utilizadas no df
    df_data_set.drop(
        columns=[
            'Total',
            # 'População residente\nem domicílios\nparticulares\npermanentes',
            # 'Homem\n(A) VALOR MEDIO',
            # 'Mulher\n(B) VALOR MEDIO',
            'Homem\n(C) MEDIANO',
            'Mulher\n(D) MEDIANO',
            'Valor médio\n(R$)',
            '1º quartil',
            '2º quartil\n(mediana)',
            '3º quartil'
        ],
        axis=1,
        inplace=True
    )
    
    df_data_set = agrupa_df_com_populacao_total(
        df_populacao=df_populacao_municipios_brasil,
        df_dataset=df_data_set
    )

    colunas_exceto = ['UF', 'MUNICIPIO']
    colunas_para_converter = [col for col in df_data_set.columns if col not in colunas_exceto]
    df_data_set[colunas_para_converter] = df_data_set[colunas_para_converter].apply(pd.to_numeric, errors='coerce')

    """ se quiser alterar os pesos dos indices, eh aqui """
    df_data_set = cria_coluna_contaminacao_indice(df_data_set)

    # normalizar colunas, tirar acentos, etc
    df_data_set = normalizar_colunas(df_data_set)

    colunas_para_renomear = {
        'COM_SANEAMENTO_INADEQUADO_(2)': 'COM_SANEAMENTO_INADEQUADO',
        'COM_RESPONSAVEL_OU_CONJUGE_ANALFABETO_E_SANEAMENTO_INADEQUADO_(2)': 'COM_RESPONSAVEL_OU_CONJUGE_ANALFABETO_E_SANEAMENTO_INADEQUADO',
        'ADEQUADO_(2)': 'ADEQUADO',
        'SEMI-ADEQUADO_(3)': 'SEMI_ADEQUADO',
        'INADEQUADO_(4)': 'INADEQUADO',
        'ATE_R$70': 'ATE_RS70',
        'ATE_1/4_SM_(=R$128)': 'ATE_SM_RS128',
        'ATE_1/2_SM_(=R$255)': 'ATE_SM_RS255',
        'ATE_60%_DA_MEDIANA_(=R$225)': 'ATE_60PC_DA_MEDIANA_RS225',
        'ATE_70,00_R$': 'ATE_70_RS',
        'ATE_1/4_SALARIO_MINIMO_(=_127,50_R$)': 'ATE_SALARIO_MINIMO_127_50_RS',
        'ATE_1/2_SALARIO_MINIMO_(=_255,00_R$)': 'ATE_SALARIO_MINIMO_255_00_RS',
        'ATE_60%_DA_MEDIANA_0_BRASIL_TOTAL_(=_225,00_R$)': 'ATE_60PC_DA_MEDIANA_0_BRASIL_TOTAL_225_00_RS',
        'BRANCA/PRETA_(A/B)': 'BRANCA_PRETA_A_B',
        'BRANCA/PARDA_(A/C)': 'BRANCA_PARDA_A_C',
        'BRANCA/AMARELA_(A/D)': 'BRANCA_AMARELA_A_D',
        'BRANCA/INDIGENA_(A/E)': 'BRANCA_INDIGENA_A_E',
        'PRETA/PARDA_(B/C)': 'PRETA_PARDA_B_C',
        'HOMEM_(A)_VALOR_MEDIO': 'HOMEM_A_VALOR_MEDIO',
        'MULHER_(B)_VALOR_MEDIO': 'MULHER_B_VALOR_MEDIO',
        'HOMEM_(C)_MEDIANO': 'HOMEM_C_MEDIANO',
        'MULHER_(D)_MEDIANO': 'MULHER_D_MEDIANO',
        'RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIO_(A/B)': 'RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIO_A_B',
        'RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIANO_(C/D)': 'RAZAO_ENTRE_VALOR_MEDIO_E_MEDIANO_DO_RENDIMENTO_MENSAL_TOTAL_NOMINAL_DE_HOMENS_E_MULHERES_MEDIANO_C_D',
        'VALOR_MEDIO_(R$)': 'VALOR_MEDIO_RS',
        '1O_QUARTIL': 'PRIMEIRO_QUARTIL',
        '2O_QUARTIL_(MEDIANA)': 'SEGUNDO_QUARTIL_MEDIANA',
        '3O_QUARTIL': 'TERCEIRO_QUARTIL'
    }

    # renomear colunas
    df_data_set.rename(
        columns={
            antiga: nova
            for antiga, nova in colunas_para_renomear.items()
            if antiga in df_data_set.columns
        },
        inplace=True
    )

    # dropa colunas que nao serao utilizadas no df
    df_data_set.drop(
        columns=[
            'MEDIA_PONDERADA_INDICE', 'MEDIA_COMUM_INDICE', 'CONTAMINACAO_INDICE_PONDERADA'
        ],
        axis=1,
        inplace=True
    )

    # tira o DF
    df_data_set = df_data_set[df_data_set['UF'] != 'DF']

    # escreve para CSV
    df_data_set.to_csv('data-set-populacao-media-sem-indice-normalizado-15-04-2025.csv', index=False, encoding='latin-1', sep=',')

    # caso o dataset ja tenha sido criado
    df_data_set = pd.read_csv('data-set-populacao-brasil-inteiro-indice-normalizado.csv', encoding='latin-1', sep=';')

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

print("Fim")
