import pandas as pd


class DFMunicipio:
    def __init__(
            self,
            ano: str,
            tipo_renda: str,
            descricao_renda: str,
            municipio: str,
            classes_renda: dict
    ):
        self.ano = ano  # ano dos dados coletados
        self.tipo_renda = tipo_renda  # basicamente eh o nome da planilha
        self.descricao_renda = descricao_renda  # descricao do que a planilha representa
        self.municipio = municipio  # qual eh o municipio do indicador
        self.classes_renda = classes_renda  # colunas (variaveis) do municipio em questão.
