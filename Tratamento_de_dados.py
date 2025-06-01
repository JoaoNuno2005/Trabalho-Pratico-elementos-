import pandas as pd

def carregar_dataframe_flexivel(path):
    with open(path, encoding="utf-8-sig") as f:
        linhas = [linha.strip() for linha in f if "," in linha]
    header = linhas[0].split(",")
    n_cols_header = len(header)
    dados_validos = [linha.split(",") for linha in linhas[1:] if len(linha.split(",")) == n_cols_header]
    df = pd.DataFrame(dados_validos, columns=header)
    return df

def carregar_datasets():
    df_desemprego = carregar_dataframe_flexivel("Desemprego.csv")
    df_ganhos = carregar_dataframe_flexivel("Ganho_medio_mensal.csv")
    df_populacao = carregar_dataframe_flexivel("populacao_residente.csv")
    return df_desemprego, df_ganhos, df_populacao

def remover_linhas_sem_regiao(df):
    col_regiao = [c for c in df.columns if "Região" in c or "Regi" in c][0]
    return df[df[col_regiao].str.strip() != ""]

def preparar_df(df, nome_valor, filtro_col=None, filtro_valor="Total"):
    df.columns = [c.strip() for c in df.columns]
    if filtro_col and filtro_col in df.columns:
        df = df[df[filtro_col].str.strip() == filtro_valor]
    col_ano = [c for c in df.columns if "Ano" in c][0]
    col_regiao = [c for c in df.columns if "Região" in c or "Regi" in c][0]
    col_valor = df.columns[-1]
    df = df.rename(columns={col_ano: "Ano", col_regiao: "Regiao", col_valor: nome_valor})
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")
    df[nome_valor] = pd.to_numeric(df[nome_valor], errors="coerce")
    df = df[["Ano", "Regiao", nome_valor]].dropna()
    return df
