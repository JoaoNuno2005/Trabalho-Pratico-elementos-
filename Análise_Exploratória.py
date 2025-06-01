# --- Importação de Bibliotecas ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from Tratamento_de_dados import carregar_datasets, remover_linhas_sem_regiao, preparar_df

# --- Carregamento e Preparação dos Dados ---
# Carregamento dos datasets
df_desemprego, df_salarios, df_populacao = carregar_datasets()

# Remoção de linhas sem informação de região
df_desemprego = remover_linhas_sem_regiao(df_desemprego)
df_salarios = remover_linhas_sem_regiao(df_salarios)
df_populacao = remover_linhas_sem_regiao(df_populacao)

# Preparação dos datasets com colunas relevantes e filtragem
df_desemprego = preparar_df(df_desemprego, "Desemprego", filtro_col="04. Filtro 1")
df_salarios = preparar_df(df_salarios, "Salario_Medio")
df_populacao = preparar_df(df_populacao, "Populacao", filtro_col="06. Filtro 2")

# Junção dos datasets num único DataFrame
df = pd.merge(df_desemprego, df_salarios, on=["Ano", "Regiao"], how="inner")
df = pd.merge(df, df_populacao, on=["Ano", "Regiao"], how="inner")

# --- Análise Exploratória Inicial ---
print("\nResumo estatístico:")
print(df.describe())

# Mapa de calor das correlações
sns.heatmap(df[["Desemprego", "Salario_Medio", "Populacao"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlações entre variáveis")
plt.show()

# Boxplots para identificar outliers
for col in ["Desemprego", "Salario_Medio", "Populacao"]:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# --- Limpeza de Outliers ---
Q1 = df[["Desemprego", "Salario_Medio", "Populacao"]].quantile(0.25)
Q3 = df[["Desemprego", "Salario_Medio", "Populacao"]].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df[["Desemprego", "Salario_Medio", "Populacao"]] < (Q1 - 1.5 * IQR)) |
                (df[["Desemprego", "Salario_Medio", "Populacao"]] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df_clean.head())
print(f"\nNúmero de dados antes da limpeza: {len(df)}")
print(f"Número de dados após a limpeza: {len(df_clean)}")

# --- Normalização dos Dados ---
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean[["Desemprego", "Salario_Medio", "Populacao"]])
df_scaled = pd.DataFrame(df_scaled, columns=["Desemprego", "Salario_Medio", "Populacao"])
df_scaled["Ano"] = df_clean["Ano"].values

# --- Clustering com KMeans e Visualização com t-SNE ---
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(df_scaled[["Desemprego", "Salario_Medio", "Populacao"]])
df_scaled["Cluster"] = labels

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_result = tsne.fit_transform(df_scaled[["Desemprego", "Salario_Medio", "Populacao", "Ano"]])
df_tsne = pd.DataFrame(tsne_result, columns=["TSNE_1", "TSNE_2"])
df_tsne["Cluster"] = df_scaled["Cluster"]

# --- Cálculo dos Centróides em TSNE ---
centroides_tsne = df_tsne.groupby("Cluster")[["TSNE_1", "TSNE_2"]].mean().reset_index()

# Visualização com t-SNE (com centróides)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_tsne, x="TSNE_1", y="TSNE_2", hue="Cluster", palette="Set1", s=60)
plt.title("Visualização dos Clusters com t-SNE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# --- Dispersão: População vs Salário Médio (com centróides) ---
centroides_pop = df_scaled.groupby("Cluster")[["Populacao", "Salario_Medio"]].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scaled, x="Populacao", y="Salario_Medio", hue="Cluster", palette="Set2", s=60)
plt.scatter(centroides_pop["Populacao"], centroides_pop["Salario_Medio"],
            c='black', s=200, marker='X', label='Centróides')
plt.title("População vs Salário Médio por Cluster com Centróides")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Análise por Região ---
# Cálculo do rendimento per capita
df_clean["Rendimento_per_capita"] = df_clean["Salario_Medio"] / df_clean["Populacao"]

# Agrupamento por região
df_regioes = df_clean.groupby("Regiao")[["Desemprego", "Salario_Medio", "Rendimento_per_capita"]].mean().reset_index()

# Normalização e Clustering
df_scaled = StandardScaler().fit_transform(df_regioes[["Desemprego", "Salario_Medio", "Rendimento_per_capita"]])
df_regioes_scaled = pd.DataFrame(df_scaled, columns=["Desemprego", "Salario_Medio", "Rendimento_per_capita"])
df_regioes_scaled["Regiao"] = df_regioes["Regiao"]
df_regioes_scaled["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(df_regioes_scaled[["Desemprego", "Salario_Medio", "Rendimento_per_capita"]])

# Visualização dos clusters por região
sns.pairplot(df_regioes_scaled, hue="Cluster", diag_kind="kde")
plt.suptitle("Clusters por Região com Salário Médio, Desemprego e Rendimento per Capita", y=1.02)
plt.tight_layout()
plt.show()

# Resumo por cluster
resumo_cluster = df_regioes_scaled.groupby("Cluster")[["Desemprego", "Salario_Medio", "Rendimento_per_capita"]].mean()
print("\nResumo por Cluster:")
print(resumo_cluster)

# --- Análises Adicionais ---
# Heatmap por cluster
plt.figure(figsize=(8, 5))
sns.heatmap(resumo_cluster, annot=True, cmap="YlGnBu")
plt.title("Média dos Indicadores por Cluster")
plt.tight_layout()
plt.show()

# Regressão: Salário Médio vs Desemprego
plt.figure(figsize=(10, 6))
sns.regplot(data=df_regioes_scaled, x="Desemprego", y="Salario_Medio", scatter_kws={'alpha':0.5})
plt.title("Regressão: Salário Médio vs Desemprego")
plt.tight_layout()
plt.show()

# Para a visualização com clusters, podemos usar scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_regioes_scaled, x="Desemprego", y="Salario_Medio", hue="Cluster")
plt.title("Salário Médio vs Desemprego por Cluster")
plt.tight_layout()
plt.show()


# --- Exportação dos Dados ---
df_clean.to_csv("dados_processados.csv", index=False)
print("Dados finais guardados em 'dados_processados.csv'")