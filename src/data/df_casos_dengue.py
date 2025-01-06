
class DFCasosDengue:
    def __init__(
            self,
            dmun_codibge: str,
            municipio_residencia: str,
            ano_epidemiologica: str,
            semana_epidemiologica: str,
            data_sintomas: str,
            data_notificacao: str,
            classificacao: str,
            criterio_confirmacao: str,
            evolucao: str
    ):
        self.dmun_codibge = dmun_codibge
        self.municipio_residencia = municipio_residencia
        self.ano_epidemiologica = ano_epidemiologica
        self.semana_epidemiologica = semana_epidemiologica
        self.data_sintomas = data_sintomas
        self.data_notificacao = data_notificacao
        self.classificacao = classificacao
        self.criterio_confirmacao = criterio_confirmacao
        self.evolucao = evolucao


