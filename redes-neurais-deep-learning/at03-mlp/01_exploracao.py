"""
01_exploracao.py
Análise exploratória do dataset WDBC (Wisconsin Diagnostic Breast Cancer),
obtido do OpenML (data_id=1510).

Saídas:
  - resumo no terminal (dimensões, tipos, faltantes, distribuição das classes)
  - resultados/estatisticas.csv (estatísticas descritivas)
  - figuras/distribuicao_classes.png
  - figuras/boxplot_escalas.png
  - figuras/mapa_correlacao.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: apenas salva as figuras em arquivo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml

# Garante acentuação correta no terminal do Windows
sys.stdout.reconfigure(encoding="utf-8")

# Pastas de saída (relativas à pasta do script)
BASE = Path(__file__).resolve().parent
DIR_RESULTADOS = BASE / "resultados"
DIR_FIGURAS = BASE / "figuras"
DIR_RESULTADOS.mkdir(exist_ok=True)
DIR_FIGURAS.mkdir(exist_ok=True)

# No OpenML os atributos vêm como V1..V30. Abaixo estão os nomes originais,
# na mesma ordem: 10 medidas x (média, erro padrão, pior valor).
MEDIDAS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
NOMES = [f"{m}_{sufixo}" for sufixo in ("mean", "se", "worst") for m in MEDIDAS]

# Rótulos da classe no OpenML: 1 = benigno, 2 = maligno
ROTULOS_CLASSE = {"1": "Benigno", "2": "Maligno"}


def secao(titulo):
    """Imprime um cabeçalho para separar as partes da saída."""
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)


# ------------------------------------------------------------------
# 1. Carregamento dos dados
# ------------------------------------------------------------------
dados = fetch_openml(data_id=1510, as_frame=True)
X = dados.data.copy()
y = dados.target.astype(str).map(ROTULOS_CLASSE)

# Renomeia V1..V30 para os nomes descritivos
X.columns = NOMES

# ------------------------------------------------------------------
# 2. Visão geral: dimensões, tipos e valores faltantes
# ------------------------------------------------------------------
secao("VISÃO GERAL")
print(f"Nome do dataset:     {dados.details.get('name')}")
print(f"Nº de amostras:      {X.shape[0]}")
print(f"Nº de atributos:     {X.shape[1]}")
print(f"Variável alvo:       {dados.target.name}")

print("\nTipos de dados dos atributos:")
print(X.dtypes.value_counts().to_string())

faltantes = X.isna().sum()
print(f"\nValores faltantes nos atributos: {int(faltantes.sum())}")
print(f"Valores faltantes no alvo:       {int(y.isna().sum())}")
if faltantes.sum() > 0:
    print(faltantes[faltantes > 0].to_string())

# ------------------------------------------------------------------
# 3. Distribuição das classes
# ------------------------------------------------------------------
secao("DISTRIBUIÇÃO DAS CLASSES")
contagem = y.value_counts()
percentual = y.value_counts(normalize=True) * 100
dist = pd.DataFrame({"contagem": contagem, "percentual (%)": percentual.round(2)})
print(dist.to_string())

# ------------------------------------------------------------------
# 4. Estatísticas descritivas
# ------------------------------------------------------------------
secao("ESTATÍSTICAS DESCRITIVAS")
estatisticas = X.agg(["mean", "std", "min", "max"]).T
estatisticas.columns = ["media", "desvio_padrao", "minimo", "maximo"]
estatisticas.index.name = "atributo"
estatisticas.to_csv(DIR_RESULTADOS / "estatisticas.csv", float_format="%.6f")

with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 120):
    print(estatisticas.to_string())
print(f"\nSalvo em: {DIR_RESULTADOS / 'estatisticas.csv'}")

# Destaca a diferença de escala entre os atributos (motivo para padronizar antes da MLP)
amplitude_max = estatisticas["maximo"].idxmax()
amplitude_min = estatisticas["maximo"].idxmin()
print(
    f"\nMaior valor máximo: {amplitude_max} = {estatisticas.loc[amplitude_max, 'maximo']:,.2f}"
    f"\nMenor valor máximo: {amplitude_min} = {estatisticas.loc[amplitude_min, 'maximo']:,.4f}"
)

# ------------------------------------------------------------------
# 5. Gráficos
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
COR = "#4C72B0"  # cor única: as classes já são identificadas pelo eixo x

# 5.1 Distribuição das classes (barras com contagem e % escritas)
fig, ax = plt.subplots(figsize=(6, 4.5))
barras = ax.bar(contagem.index, contagem.values, color=COR, width=0.55)
for barra, n, p in zip(barras, contagem.values, percentual.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2, n + 5,
        f"{n} ({p:.1f}%)", ha="center", va="bottom", fontsize=11,
    )
ax.set_title("Distribuição das classes - WDBC")
ax.set_xlabel("Diagnóstico")
ax.set_ylabel("Nº de amostras")
ax.set_ylim(0, contagem.max() * 1.15)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "distribuicao_classes.png", dpi=150)
plt.close(fig)

# 5.2 Boxplot de alguns atributos para evidenciar as escalas diferentes.
# Painel da esquerda em escala linear (area domina e os outros "somem");
# painel da direita em escala log, onde todos ficam visíveis.
atributos_box = [
    "area_mean", "perimeter_mean", "texture_mean",
    "radius_mean", "smoothness_mean", "fractal_dimension_mean",
]
longo = X[atributos_box].melt(var_name="atributo", value_name="valor")

fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
for ax, escala in zip(eixos, ["linear", "log"]):
    sns.boxplot(data=longo, x="atributo", y="valor", color=COR, ax=ax,
                fliersize=2, linewidth=1)
    ax.set_yscale(escala)
    ax.set_title(f"Escala {escala}")
    ax.set_xlabel("")
    ax.set_ylabel("Valor" + (" (log)" if escala == "log" else ""))
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Atributos em escalas muito diferentes (justifica padronização)")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "boxplot_escalas.png", dpi=150)
plt.close(fig)

# 5.3 Mapa de correlação (Pearson) entre os 30 atributos
corr = X.corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
    linewidths=0.3, cbar_kws={"shrink": 0.7, "label": "Correlação de Pearson"}, ax=ax,
)
ax.set_title("Mapa de correlação entre os atributos - WDBC")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "mapa_correlacao.png", dpi=150)
plt.close(fig)

# Lista os pares mais correlacionados (útil para discutir redundância)
secao("PARES MAIS CORRELACIONADOS (|r| > 0.95)")
pares = (
    corr.where(~pd.DataFrame(
        [[i >= j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns,
    ))
    .stack()
    .sort_values(key=abs, ascending=False)
)
print(pares[pares.abs() > 0.95].round(3).to_string())

secao("FIGURAS SALVAS")
for arquivo in sorted(DIR_FIGURAS.glob("*.png")):
    print(f"  {arquivo}")
