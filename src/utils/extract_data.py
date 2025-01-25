import pandas as pd
from pandas import DataFrame
import unicodedata

from src.data.build_df import BuildDf


def normalizar_string(text):
    try:
        if int(text):
            numero = pd.to_numeric(text.replace(" ", ""), errors='coerce')
            return numero
    except Exception as error:
        raise Exception(f"Erro ao tirar espaços em branco do numero {text}! {error}")
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()
    except Exception as error:
        raise Exception(f"Erro ao normalizar a string {text}! {error}")


def obtem_dados_municipio(df: DataFrame, municipio: str, colunas: list):
    coluna_municipio = None
    municipio_normalizado = ''
    nome_padronizado_coluna = 'municipio'

    for coluna in colunas:
        try:
            municipio_normalizado = normalizar_string(municipio)
            df_municipio_normalizado = df[coluna].apply(normalizar_string)
        except Exception as error:
            raise Exception(f"Erro ao normalizar municipio e coluna de municipio do dataframe! {error}")

        if municipio_normalizado in df_municipio_normalizado.values:
            coluna_municipio = nome_padronizado_coluna  # padronizacao dos nomes de colunas de municipio.
            df = df.rename(columns={coluna: coluna_municipio})
            df[coluna_municipio] = df[coluna_municipio].apply(normalizar_string)
            break

    if coluna_municipio:
        linha_municipio = df[df[coluna_municipio] == municipio_normalizado]
    else:
        raise ModuleNotFoundError(f"Municipio nao encontrado no df!")

    return linha_municipio
