# monta os DFs
import pandas as pd

from src.data.build_df import BuildDF
from src.utils.extract_data import obtem_dados_municipio
from src.utils.load_file import get_sheet_names

# arquivos XLSX com datasets
xlsx_casos_dengue = 'G:\\Meu Drive\\MBA-DATA-SCIENCE\\TCC\\dados\\GOIANIA_2010_dengue-casos-notificados.xlsx'
xlsx_municipios = 'G:\\Meu Drive\\MBA-DATA-SCIENCE\\TCC\\dados\\DADOS-INDICADORES-SOCIOECONOMICOS\\DATA-SET-2010.xlsx'
municipio_interesse = "Goiânia"

# obtem nomes das planilhas do xlsx
sheet_names_casos_dengue = get_sheet_names(xlsx_casos_dengue)
sheet_names_municipios = get_sheet_names(xlsx_municipios)

lista_dfs_municipios = []
lista_dfs_casos_dengue = []

for sheet_municio in sheet_names_municipios:
    df_municipio = BuildDF(
        xlsx_name=xlsx_municipios,
        sheet_name=sheet_municio,
        skiprows=1
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
    df_dengue = BuildDF(
        xlsx_name=xlsx_casos_dengue,
        sheet_name=sheet_municio,
        skiprows=None
    )
    lista_dfs_casos_dengue.append(df_dengue)
    del df_dengue

print("Fim")
