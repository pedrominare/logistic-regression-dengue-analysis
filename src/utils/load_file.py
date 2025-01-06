import pandas as pd


def create_df_from_xlsx(file_name, sheet_name, skiprows=None):
    try:
        if skiprows:
            df = pd.read_excel(file_name, engine='openpyxl', sheet_name=sheet_name, skiprows=skiprows)
        else:
            # Lê o arquivo em chunks de 100.000 linhas
            df = pd.read_excel(file_name, engine='openpyxl', sheet_name=sheet_name)
    except Exception as error:
        raise FileExistsError(f"Erro ao tentar carregar o arquivo {file_name}! {error}")

    return df


def get_sheet_names(file_name):
    try:
        excel_file = pd.ExcelFile(file_name)
        sheet_names = excel_file.sheet_names
    except Exception as error:
        raise FileExistsError(f"Erro ao tentar carregar os nomes das planilhas do arquivo {file_name}! {error}")

    return sheet_names
