import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd

from data.municipios import estados_ibge
from utils.df_helper import normalizar_string


class WebScraping:
    def __init__(self):
        self.sessao = requests.session()

    def obtem_populacao_estado(self):
        lista_dfs_municipios = []

        for uf, cod_uf in estados_ibge.items():
            response_pagina_municipios = self.sessao.get(
                url=f"https://censo2010.ibge.gov.br/sinopse/index.php?uf={cod_uf}&dados=1",
                headers={
                    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                    "accept-language": "pt-BR,pt;q=0.9",
                    "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "sec-fetch-dest": 'document',
                    "sec-fetch-mode": 'navigate',
                    "sec-fetch-site": 'none',
                    'sec-fetch-user': '?1',
                    'upgrade-insecure-requests': '1',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
                },
            )

            if response_pagina_municipios.status_code != 200 or 'Total' not in response_pagina_municipios.text:
                raise Exception(f"Site retornou statuscode {response_pagina_municipios.status_code} e o conteudo da pagina nao foi o esperado! {response_pagina_municipios.text}")

            try:
                strainer = SoupStrainer(['table', 'tr'])
                soup = BeautifulSoup(response_pagina_municipios.text, 'html.parser', parse_only=strainer, from_encoding='ISO-8559-1')
                tabela_municipios = soup.find("table", attrs={'class': 'tabela_resultados sortable'})
                linhas = tabela_municipios.find_all('tr')
            except Exception as error:
                raise Exception(f"Erro no parser do HTML! {error}")

            lista_dados_municipios = []
            cabecalhos = linhas[0].find_all('th', attrs={'class': 'th_ordenavel'})
            if cabecalhos is None:
                raise Exception("cabecalhos nao encontrados!")

            linhas.pop(0) # remove o cabecalho para o for a seguir
            for linha in linhas:
                linha_dados = linha.find_all('td')
                dados_municipios = {
                    "UF": uf
                }
                #dados_municipios["UF"] = uf

                for i in range(0, 11):
                    chave = normalizar_string(cabecalhos[i].text)
                    valor = normalizar_string(linha_dados[i].text.replace(".", "").replace("\n", ""))
                    dados_municipios[chave] = valor

                lista_dados_municipios.append(dados_municipios)
                del dados_municipios

            if len(lista_dados_municipios) == 0:
                raise Exception("Lista de dados de populacao total dos municipios vazia!")

            lista_dfs_municipios.append(pd.DataFrame(lista_dados_municipios))

        return pd.concat(lista_dfs_municipios).fillna(0)
