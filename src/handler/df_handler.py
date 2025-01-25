import re

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

        def remover_ponto(texto):
            return re.sub(r'\.', '', texto)

        def validar_numero(texto):
            contem_numero = any(caractere.isdigit() for caractere in texto)
            return contem_numero

        for item_municipio_interesse in lista_municipios_interesse:
            municipio_interesse = normalizar_string(item_municipio_interesse)

            if validar_numero(municipio_interesse):
                continue

            lista_dados_municipios = []
            for df_municipio in self.municipio_obj_list:
                for item in df_municipio.variables:
                    try:
                        item_normalizado = normalizar_string(item)
                    except Exception as error:
                        raise Exception(f"Erro ao normalizar a string {item}. {error}")

                    if variavel_municipio in item_normalizado:
                        # pega dados do municipio especifico
                        dados_municipio = df_municipio.df[
                            df_municipio.df[item].apply(normalizar_string) == municipio_interesse
                        ]
                        # exclui a coluna original de municipio
                        dados_municipio = dados_municipio.drop(columns=[item])

                        # converte os valores para float
                        dados_municipio = dados_municipio.map(lambda x: str(x).replace(' ', ''))
                        dados_municipio = dados_municipio.astype(float, errors='ignore')

                        # limpa ponto como caractere especial do nome do municipio
                        municipio_interesse = remover_ponto(municipio_interesse)

                        # cria uma coluna municipio
                        dados_municipio[variavel_municipio] = municipio_interesse
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
                df_municipios = df_municipios.groupby(variavel_municipio).max()
            except Exception as error:
                raise Exception(f"Erro ao agrupar a lista de dados de municipios! {error}")

            try:
                df_municipios = df_municipios.reset_index(drop=False).fillna(0)
            except Exception as error:
                raise Exception(f"Erro ao recuperar a coluna MUNICIPIO dos dfs lista de dados de municipios!! {error}")
            lista_dfs_municipios.append(df_municipios)
        try:
            self.municipios_concatenados = pd.concat(lista_dfs_municipios)
        except Exception as error:
            raise Exception(f"Erro ao concatenar dfs! {error}")

    def agrupa_casos_dengue(self, casos: list):
        for df_dengue in self.dengue_obj_list:
            df_filtrado = df_dengue.df[df_dengue.df['CLASSIFICACAO'].isin(casos)]
            df_agrupado = df_filtrado.groupby('MUNICIPIO')['CLASSIFICACAO'].value_counts().unstack(fill_value=0)
            df_agrupado['DENGUE'] = df_agrupado[casos[0]] + df_agrupado[casos[1]]
            return df_agrupado.groupby('MUNICIPIO')['DENGUE'].sum().reset_index()

    def agrupa_casos_dengue_com_municipios(self, df_somatorio_dengue):
        df_dataset = pd.merge(self.municipios_concatenados, df_somatorio_dengue, on='MUNICIPIO', how='left').fillna(0)
        df_dataset.to_csv('data-set-pronto.csv', index=False, encoding='latin-1', sep=';')

        return df_dataset
