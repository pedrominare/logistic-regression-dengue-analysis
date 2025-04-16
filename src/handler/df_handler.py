import re
import unidecode

from src.data.build_df import BuildDf
from src.utils.df_helper import get_sheet_names, cross_data
from utils.df_helper import normalizar_string
from functools import reduce

import pandas as pd
from pandas import DataFrame


class DfHandler(BuildDf):
    def __init__(self, xlsx_name, logging, memory_usage, skiprows=0, ano="", chunk_size=1):
        super().__init__(xlsx_name=xlsx_name, sheet_name="", logging=logging, skiprows=skiprows, chunk_size=chunk_size, memory_usage=memory_usage)
        self.ano = ano
        self.municipio_obj_list = []
        self.df_municipio_crossed = None
        self.dengue_obj_list = []
        self.df_concatenado = None
        self.municipios_concatenados = None
        self.dependent_variable = "CLASSIFICACAO"

    def build_df(self, tipo: str):
        # obtem nomes das planilhas do xlsx
        # o nome da planilha de contaminacao de dengue deve corresponder ao ano dos registros.
        sheet_names_xlsx = get_sheet_names(self.xlsx_name)

        for sheet in sheet_names_xlsx:
            if "dengue" in tipo.lower():
                if self.ano in sheet:
                    obj_df = BuildDf(
                        xlsx_name=self.xlsx_name,
                        sheet_name=sheet,
                        logging=self.logging,
                        skiprows=self.skiprows,
                        chunk_size=1000,
                        memory_usage=self.memory_usage
                    )
                else:
                    continue
            elif "municipio" in tipo.lower():
                obj_df = BuildDf(
                    xlsx_name=self.xlsx_name,
                    sheet_name=sheet,
                    logging=self.logging,
                    skiprows=self.skiprows,
                    chunk_size=20,
                    memory_usage=self.memory_usage
                )
            else:
                raise Exception(f"Não foi identificado o tipo [{tipo}] de DF a ser criado!")

            # gera o DF do arquivo
            obj_df.build_data_frame()
            self.memory_usage.add_checkpoint(name=f"DFs de {sheet} construidos")

            # carrega as variaveis do DF criado
            obj_df.get_variables()

            if "municipio" in tipo.lower():
                self.municipio_obj_list.append(obj_df)
            elif "dengue" in tipo.lower():
                self.dengue_obj_list.append(obj_df)
            else:
                raise Exception(f"Não foi identificado o tipo [{tipo}] de DF a ser criado!")

    def cria_lista_df_municipios(self, lista_municipios_interesse):
        lista_dfs_municipios = []
        variavel_municipio = "MUNICIPIO"
        variavel_uf = "UF"

        def remover_ponto(texto):
            return re.sub(r'\.', '', texto)

        def validar_numero(texto):
            contem_numero = any(caractere.isdigit() for caractere in texto)
            return contem_numero

        for item_dict in lista_municipios_interesse:
            for uf_interesse_, municipio_interesse_ in item_dict.items():
                municipio_interesse = normalizar_string(municipio_interesse_)
                uf_interesse = normalizar_string(uf_interesse_)

                if validar_numero(municipio_interesse):
                    continue

                lista_dados_municipios = []
                for df_municipio in self.municipio_obj_list:
                    for uf, municipio in zip(df_municipio.variables[::2], df_municipio.variables[1::2]):
                    # for item in df_municipio.variables:
                        try:
                            municipio_normalizado = normalizar_string(municipio)
                        except Exception as error:
                            raise Exception(f"Erro ao normalizar a string {municipio}. {error}")

                        if variavel_municipio in municipio_normalizado:
                            # pega dados do municipio especifico
                            dados_municipio = df_municipio.df[
                                (df_municipio.df[municipio].apply(normalizar_string) == municipio_interesse) &
                                (df_municipio.df[uf].apply(normalizar_string) == uf_interesse)
                            ]
                            # exclui a coluna original de municipio
                            dados_municipio = dados_municipio.drop(columns=[municipio, uf])

                            # converte os valores para float
                            dados_municipio = dados_municipio.map(lambda x: str(x).replace(' ', ''))
                            dados_municipio = dados_municipio.astype(float, errors='ignore')

                            # limpa ponto como caractere especial do nome do municipio
                            municipio_interesse = remover_ponto(municipio_interesse)

                            # cria uma coluna municipio
                            dados_municipio[variavel_municipio] = municipio_interesse
                            dados_municipio[variavel_uf] = uf_interesse
                            # limpa dados None e NaN
                            dados_municipio = dados_municipio.dropna(axis=1)
                            # limpar colunas None
                            dados_municipio = dados_municipio.drop(
                                columns=[
                                    col for col in dados_municipio.columns if not col or pd.isna(col)
                                ]
                            )
                            lista_dados_municipios.append(dados_municipio)

                try:
                    df_municipios = pd.concat(lista_dados_municipios).fillna(0)
                except Exception as error:
                    raise Exception(f"Erro ao concatenar a lista de dados de municipios! {error}")

                try:
                    df_municipios = df_municipios.groupby(variavel_uf).max()
                except Exception as error:
                    raise Exception(f"Erro ao agrupar a lista de dados de municipios! {error}")

                try:
                    df_municipios = df_municipios.reset_index(drop=False).fillna(0)
                except Exception as error:
                    raise Exception(f"Erro ao recuperar a coluna MUNICIPIO dos dfs lista de dados de municipios!! {error}")
                lista_dfs_municipios.append(df_municipios)

            #lista_municipios_interesse.remove(item_dict)
        try:
            self.municipios_concatenados = pd.concat(lista_dfs_municipios)
            self.municipios_concatenados.to_csv('data-set-completo-municipios-brasil.csv', index=False, encoding='latin-1', sep=';')
        except Exception as error:
            raise Exception(f"Erro ao concatenar dfs! {error}")

    def cria_lista_df_municipios_otimizado(self, lista_municipios_interesse):
        lista_dfs_municipios = []
        variavel_municipio = "MUNICIPIO"
        variavel_uf = "UF"

        def remover_ponto(texto):
            return re.sub(r'\.', '', texto)

        def validar_numero(texto):
            contem_numero = any(caractere.isdigit() for caractere in texto)
            return contem_numero

        for item_dict in lista_municipios_interesse:
            # cada uf: municipio é um item_dict
            for uf_interesse_, municipio_interesse_ in item_dict.items():
                municipio_interesse = normalizar_string(municipio_interesse_)
                uf_interesse = normalizar_string(uf_interesse_)

                if validar_numero(municipio_interesse):
                    continue

                lista_dados_municipios = []
                for df_municipio in self.municipio_obj_list:
                    for uf, municipio in zip(df_municipio.variables[::2], df_municipio.variables[1::2]):
                    # for item in df_municipio.variables:
                        try:
                            municipio_normalizado = normalizar_string(municipio)
                        except Exception as error:
                            raise Exception(f"Erro ao normalizar a string {municipio}. {error}")

                        if variavel_municipio in municipio_normalizado:
                            # pega dados do municipio especifico
                            dados_municipio = df_municipio.df[
                                (df_municipio.df[municipio].apply(normalizar_string) == municipio_interesse) &
                                (df_municipio.df[uf].apply(normalizar_string) == uf_interesse)
                            ]
                            # exclui a coluna original de municipio
                            dados_municipio = dados_municipio.drop(columns=[municipio, uf])

                            # converte os valores para float
                            dados_municipio = dados_municipio.map(lambda x: str(x).replace(' ', ''))
                            dados_municipio = dados_municipio.astype(float, errors='ignore')

                            # limpa ponto como caractere especial do nome do municipio
                            municipio_interesse = remover_ponto(municipio_interesse)

                            # cria uma coluna municipio
                            dados_municipio[variavel_municipio] = municipio_interesse
                            dados_municipio[variavel_uf] = uf_interesse
                            # limpa dados None e NaN
                            dados_municipio = dados_municipio.dropna(axis=1)
                            # limpar colunas None
                            dados_municipio = dados_municipio.drop(
                                columns=[
                                    col for col in dados_municipio.columns if not col or pd.isna(col)
                                ]
                            )
                            lista_dados_municipios.append(dados_municipio)
                            break

                try:
                    df_municipios = pd.concat(lista_dados_municipios).fillna(0)
                except Exception as error:
                    raise Exception(f"Erro ao concatenar a lista de dados de municipios! {error}")

                try:
                    df_municipios = df_municipios.groupby(variavel_uf).max()
                except Exception as error:
                    raise Exception(f"Erro ao agrupar a lista de dados de municipios! {error}")

                try:
                    df_municipios = df_municipios.reset_index(drop=False).fillna(0)
                except Exception as error:
                    raise Exception(f"Erro ao recuperar a coluna MUNICIPIO dos dfs lista de dados de municipios!! {error}")
                lista_dfs_municipios.append(df_municipios)

            #lista_municipios_interesse.remove(item_dict)
        try:
            self.municipios_concatenados = pd.concat(lista_dfs_municipios)
            self.municipios_concatenados.to_csv('data-set-completo-municipios-brasil.csv', index=False, encoding='latin-1', sep=';')
        except Exception as error:
            raise Exception(f"Erro ao concatenar dfs! {error}")

    def agrupa_casos_dengue(self, casos: list):
        for df_dengue in self.dengue_obj_list:
            df_filtrado = df_dengue.df[df_dengue.df['CLASSIFICACAO'].isin(casos)]
            df_agrupado = df_filtrado.groupby('MUNICIPIO')['CLASSIFICACAO'].value_counts().unstack(fill_value=0)
            df_agrupado['DENGUE'] = df_agrupado[casos[0]] + df_agrupado[casos[1]]
            return df_agrupado.groupby('MUNICIPIO')['DENGUE'].sum().reset_index()

    def agrupa_casos_dengue_brasil_inteiro(self):
        for df_dengue in self.dengue_obj_list:
            # df_filtrado = df_dengue.df[df_dengue.df['CLASSIFICACAO'].isin(casos)]
            df_agrupado = (
                df_dengue.groupby(
                    'MUNICIPIO',
                    'UF'
                )
            )
            # df_agrupado['DENGUE'] = df_agrupado[casos[0]] + df_agrupado[casos[1]]
            # return df_agrupado.groupby('MUNICIPIO')['DENGUE'].sum().reset_index()
            return df_agrupado

    def agrupa_casos_dengue_com_municipios(self, df_somatorio_dengue):
        df_dataset = pd.merge(self.municipios_concatenados, df_somatorio_dengue, on='MUNICIPIO', how='left').fillna(0)
        df_dataset.to_csv('data-set-pronto.csv', index=False, encoding='latin-1', sep=';')

        return df_dataset

    def agrupa_casos_dengue_com_municipios_brasil_todo(self, obj_dengue):
        lista_dfs_dengue = []
        for df_dengue in obj_dengue.dengue_obj_list:
            df_dataset = pd.merge(self.municipios_concatenados, df_dengue.df, on=['MUNICIPIO', 'UF'], how='left').fillna(0)
            lista_dfs_dengue.append(df_dataset)

        df_resultado_agrupamento = pd.concat(lista_dfs_dengue)
        df_resultado_agrupamento.to_csv('data-set-pronto-dengue-municipios.csv', index=False, encoding='latin-1', sep=';')

        return df_resultado_agrupamento


def agrupa_df_com_populacao_total(df_populacao, df_dataset):
    df_dataset_populacao = pd.merge(df_populacao, df_dataset, on=['MUNICIPIO', 'UF'], how='left').fillna(0)
    # df_dataset_populacao.to_csv('data-set-populacao.csv', index=False, encoding='latin-1', sep=';')

    return df_dataset_populacao


def normalizar_colunas(df):
    df.columns = [
        unidecode.unidecode(col)  # Remove acentos
        .replace("\n", "_")  # Substitui quebras de linha por _
        .replace(" ", "_")  # Substitui espaços por _
        .upper()  # Converte para maiusculas
        for col in df.columns
    ]
    return df


def cria_coluna_contaminacao_indice(df_dataset):
    # media ponderada dos indices socioeconomicos
    lista_indices_socioeconomicos = {
        "p1": {
            "indice": 'Proporção_de_crianças_de_0_a_5_anos_residentes_em_domicilios_particulares_permanentes_com_responsável_ou_cônjuge_analfabeto',
            'peso': 1
        },
        "p2": {
            "indice": 'Proporção_de_crianças_de_0_a_5_anos_residentes_em_domicilios_particulares_permanentes_com_saneamento_inadequado',
            'peso': 1.10
        },
        "p3": {
            "indice": 'Proporção_de_crianças_de_0_a_5_anos_residentes_em_domicilios_particulares_permanentes_com_responsável_ou_cônjuge_analfabeto_e_saneamento_inadequado',
            'peso': 1.10
        },
        "p4": {
            "indice": 'Proporção de domicilios particulares permanentes por tipo de saneamento\nAdequado',
            'peso': 1.10
        },
        "p5": {
            "indice": 'Proporção de domicilios particulares permanentes por tipo de saneamento\nSemi-Adequado',
            'peso': 1.15
        },
        "p6": {
            "indice": 'Proporção de domicilios particulares permanentes por tipo de saneamento\nInadequado',
            'peso': 1.20
        },
        "p7": {
            "indice": 'População residente em domicilios particulares permanentes com saneamento inadequado e rendimento mensal total domiciliar per capita nominal\nTotal',
            'peso': 1.10
        },
        "p8": {
            "indice": 'População residente em domicilios particulares permanentes com saneamento inadequado e rendimento mensal total domiciliar per capita nominal de Até R$70',
            'peso': 1.15
        },
        "p9": {
            "indice": 'População residente em domicilios particulares permanentes com saneamento inadequado e rendimento mensal total domiciliar per capita nominal de Até 1/4 SM\n(=R$128)',
            'peso': 1.10
        },
        "p10": {
            "indice": 'População residente em domicilios particulares permanentes com saneamento inadequado e rendimento mensal total domiciliar per capita nominal de \nAté 1/2 SM\n(=R$255)',
            'peso': 1.10
        },
        "p11": {
            "indice": 'População residente em domicilios particulares permanentes com saneamento inadequado e rendimento mensal total domiciliar per capita nominal de Até 60% da mediana\n(=R$225)',
            'peso': 1.10
        },
        "p12": {
            "indice": 'População residente\nem domicílios\nparticulares\npermanentes',
            'peso': 1
        },
        "p13": {
            "indice": 'Proporção de pessoas residentes em domicilios particulares permanentes com rendimento mensal total domiciliar per capita de\nAté 70,00 R$',
            'peso': 1
        },
        "p14": {
            "indice": 'Proporção de pessoas residentes em domicilios particulares permanentes com rendimento mensal total domiciliar per capita de Até 1/4 salário mínimo\n(= 127,50 R$)',
            'peso': 1
        },
        "p15": {
            "indice": 'Proporção de pessoas residentes em domicilios particulares permanentes com rendimento mensal total domiciliar per capita de Até 1/2 salário mínimo\n(= 255,00 R$)',
            'peso': 1
        },
        "p16": {
            "indice": 'Proporção de pessoas residentes em domicilios particulares permanentes com rendimento mensal total domiciliar per capita de Até 60% da mediana - Brasil total\n(= 225,00 R$)',
            'peso': 1
        },
        "p17": {
            "indice": 'Razão entre médias do rendimento mensal total nominal de pessoas\nBrancas/Pretas com 10 anos ou mais residentes em domicilios particulares permanentes\n(A/B)',
            'peso': 1
        },
        "p18": {
            "indice": 'Razão entre médias do rendimento mensal total nominal de pessoas\nBrancas/Pardas com 10 anos ou mais residentes em domicilios particulares permanentes\n(A/C)',
            'peso': 1
        },
        "p19": {
            "indice": 'Razão entre médias do rendimento mensal total nominal de pessoas\nBrancas/Amarelas com 10 anos ou mais residentes em domicilios particulares permanentes\n(A/D)',
            'peso': 1
        },
        "p20": {
            "indice": 'Razão entre médias do rendimento mensal total nominal de pessoas\nBrancas/Indigenas com 10 anos ou mais residentes em domicilios particulares permanentes\n(A/E)',
            'peso': 1
        },
        "p21": {
            "indice": 'Razão entre médias do rendimento mensal total nominal de pessoas\nPretas/Pardas com 10 anos ou mais residentes em domicilios particulares permanentes\n(B/C)',
            'peso': 1
        },
        "p22": {
            "indice": 'Valor médio do rendimento mensal total domiciliar per capita nominal\n(R$)',
            'peso': 1
        },
        "p23": {
            "indice": '1 quartil do rendimento mensal total domiciliar per capita nominal',
            'peso': 1
        },
        "p24": {
            "indice": '2 quartil\n(mediana) do rendimento mensal total domiciliar per capita nominal',
            'peso': 1
        },
        "p25": {
            "indice": '3 quartil do rendimento mensal total domiciliar per capita nominal',
            'peso': 1
        },
        "p26": {
            "indice": 'Rendimento mensal total nominal de Homens com 10 anos ou mais residentes em domicilios particulares permanentes\n(A) VALOR MEDIO',
            'peso': 1
        },
        "p27": {
            "indice": 'Rendimento mensal total nominal de Mulheres com 10 anos ou mais residentes em domicilios particulares permanentes\n(B) VALOR MEDIO',
            'peso': 1
        },
        "p28": {
            "indice": 'Rendimento mensal total nominal de Homens com 10 anos ou mais residentes em domicilios particulares permanentes\n(C) MEDIANO',
            'peso': 1
        },
        "p29": {
            "indice": 'Rendimento mensal total nominal de Mulheres com 10 anos ou mais residentes em domicilios particulares permanentes\n(D) MEDIANO',
            'peso': 1
        },
        "p30": {
            "indice": 'Razão entre valor médio e mediano do rendimento mensal total nominal de homens e mulheres MEDIO (A/B)',
            'peso': 1
        },
        "p31": {
            "indice": 'Razão entre valor médio e mediano do rendimento mensal total nominal de homens e mulheres MEDIANO (C/D)',
            'peso': 1
        },
    }

    lista_de_pesos = []
    lista_de_colunas = []

    for i in range(1, 32):
        peso_indice = lista_indices_socioeconomicos[f"p{i}"]['peso']
        indice = lista_indices_socioeconomicos[f'p{i}']["indice"]

        if indice in df_dataset.columns:
            lista_de_colunas.append(indice)
            lista_de_pesos.append(peso_indice)

    def calcula_media_ponderada(row):
        media_ponderada_indices = (row[lista_de_colunas] * lista_de_pesos).sum() / sum(lista_de_pesos)
        resultado = float((row['DENGUE'] * media_ponderada_indices) / (row['TOTAL']))

        return resultado

    def calcula_media_comum(row):
        resultado = float((row['DENGUE']) / (row['TOTAL']))
        return resultado

    try:
        # Criar a coluna CONTAMINACAO (razão entre DENGUE e TOTAL)
        # df_dataset["CONTAMINACAO"] = df_dataset["DENGUE"] / df_dataset["TOTAL"]

        df_dataset["MEDIA_PONDERADA_INDICE"] = df_dataset.apply(
            calcula_media_ponderada,
            axis=1
        )

        df_dataset["MEDIA_COMUM_INDICE"] = df_dataset.apply(
            calcula_media_comum,
            axis=1
        )

        # Criar a coluna binária baseada no limite de 50%
        df_dataset["CONTAMINACAO_INDICE_PONDERADA"] = df_dataset["MEDIA_PONDERADA_INDICE"].apply(lambda x: 1 if x >= 1 else 0)
        df_dataset["CONTAMINACAO_INDICE"] = df_dataset["MEDIA_COMUM_INDICE"].apply(lambda x: 1 if (x*100) >= 1 else 0)

        return df_dataset
    except Exception as error:
        raise Exception(f"Erro ao tentar criar a coluna CONTAMINACAO e CONTAMINACAO_INDICE! {error}")

