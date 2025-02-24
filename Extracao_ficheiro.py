import requests
from bs4 import BeautifulSoup
import pandas as pd
import tools
# URL da página com os dados
url = "https://www.pordata.pt/pt/estatisticas/economia/crescimento-e-produtividade/taxa-de-crescimento-real-do-pib"

# Definir um cabeçalho de User-Agent para evitar bloqueios
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

# Fazer a requisição HTTP
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("Erro ao acessar a página. Código de status:", response.status_code)
else:
    soup = BeautifulSoup(response.text, "html.parser")

    # Encontrar todas as tabelas na página
    tables = soup.find_all("table")
    print(f"Número de tabelas encontradas: {len(tables)}")

    if tables:
        # Selecionar a primeira tabela (ajustar se necessário)
        table = tables[0]

        # Extrair todas as linhas da tabela
        rows = table.find_all("tr")
        data = []

        for row in rows:
            cols = row.find_all(["td", "th"])  # Inclui cabeçalhos e dados
            cols = [col.text.strip() for col in cols]
            data.append(cols)

        # Criar um DataFrame
        df = pd.DataFrame(data)

        # Exibir os dados para validação
        tools.display_dataframe_to_user(name="Taxa de Crescimento do PIB", dataframe=df)

        # Salvar em CSV
        df.to_csv("Crescimento_PIB.csv", index=False, encoding="utf-8")
        print("Dados salvos com sucesso em 'Crescimento_PIB.csv'.")
    else:
        print("Nenhuma tabela encontrada na página.")


