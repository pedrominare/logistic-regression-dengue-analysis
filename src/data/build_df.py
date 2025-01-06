from src.utils.load_file import create_df_from_xlsx


class BuildDF:
    def __init__(self, xlsx_name, sheet_name, skiprows):
        self.skiprows = skiprows
        self.sheet_name = sheet_name
        self.df = create_df_from_xlsx(xlsx_name, self.sheet_name, self.skiprows)
        self.variables = None

    def get_variables(self):
        self.variables = self.df.columns.tolist()
