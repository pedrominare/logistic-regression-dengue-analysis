from src.utils.df_helper import create_df_from_large_xlsx


class BuildDf:
    def __init__(self, xlsx_name, memory_usage, sheet_name, skiprows, chunk_size, logging):
        self.xlsx_name = xlsx_name
        self.sheet_name = sheet_name
        self.skiprows = skiprows
        self.chunk_size = chunk_size
        self.df = None
        self.variables = None
        self.logging = logging
        self.memory_usage = memory_usage

    def get_variables(self):
        self.variables = self.df.columns.tolist()

    def build_data_frame(self):
        self.df = create_df_from_large_xlsx(
            file_name=self.xlsx_name,
            sheet_name=self.sheet_name,
            start_row=self.skiprows,
            chunk_size=self.chunk_size,
            logging=self.logging,
        )
