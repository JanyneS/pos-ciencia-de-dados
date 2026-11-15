"""
05_modelo_final.py
Avaliação do modelo final no conjunto de TESTE (primeira e única vez que o teste é usado).

Modelo final (escolhido nos scripts 02 a 04 com validação cruzada no treino):
  Pipeline StandardScaler + MLPClassifier((100,), adam, lr=0.001, max_iter=2000)

Saídas:
  - resultados/modelo_final.csv       (métricas no teste: semente 42, sementes 0-9, média e desvio)
  - resultados/comparacao_teste.csv   (MLP final x Regressão Logística x MLP sem normalização)
  - figuras/matriz_confusao.png
  - figuras/curva_roc.png
  - figuras/comparacao_teste.png
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem janela: apenas salva as figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import DIR_FIGURAS, DIR_RESULTADOS, carregar_dados, criar_pastas

# Garante acentuação correta no terminal do Windows
sys.stdout.reconfigure(encoding="utf-8")
criar_pastas()

SEMENTE = 42
NOMES_CLASSES = ["Benigno", "Maligno"]  # 0 = Benigno, 1 = Maligno
ACC_CV_REFERENCIA = 0.9824              # acurácia média na validação cruzada (scripts 03/04)
DESVIO_CV_REFERENCIA = 0.0115


def secao(titulo):
    """Imprime um cabeçalho para separar as partes da saída."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def mlp_final(semente=SEMENTE):
    """Modelo final: StandardScaler + MLP (100,) com Adam, lr=0.001."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), solver="adam",
                              learning_rate_init=0.001, max_iter=2000,
                              random_state=semente)),
    ])


def treinar_e_avaliar(modelo, X_tr, y_tr, X_te, y_te):
    """
    Treina no treino, avalia no teste e devolve (modelo, métricas).
    Os ConvergenceWarnings do treino são contados e entram nas métricas.
    """
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(X_tr, y_tr)
    n_avisos = sum(issubclass(w.category, ConvergenceWarning) for w in capturados)

    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)[:, 1]  # probabilidade da classe Maligno
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    # n_iter_ só existe no MLP (último passo do Pipeline ou o próprio modelo)
    estimador = modelo[-1] if isinstance(modelo, Pipeline) else modelo
    metricas = {
        "acuracia": accuracy_score(y_te, y_pred),
        "precisao": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_prob),
        "VN": tn, "FP": fp, "FN": fn, "VP": tp,  # FN = maligno classificado como benigno
        "n_iter": getattr(estimador, "n_iter_", np.nan),
        "avisos_convergencia": n_avisos,
    }
    return modelo, metricas


# ------------------------------------------------------------------
# 1. Dados e divisão treino/teste (mesma dos scripts anteriores)
# ------------------------------------------------------------------
X, y = carregar_dados(binario=True)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMENTE
)
print(f"Treino: {len(y_treino)} amostras | Teste: {len(y_teste)} amostras "
      f"({int(y_teste.sum())} malignos, {int((y_teste == 0).sum())} benignos)")

# ------------------------------------------------------------------
# 2. Modelo final (semente 42): treino completo e avaliação no teste
# ------------------------------------------------------------------
modelo, met_final = treinar_e_avaliar(mlp_final(), X_treino, y_treino, X_teste, y_teste)
y_pred = modelo.predict(X_teste)

secao("MODELO FINAL NO TESTE (random_state=42)")
for nome in ("acuracia", "precisao", "recall", "f1", "roc_auc"):
    print(f"  {nome:<10} {met_final[nome]:.4f}")
print(f"  épocas (n_iter_): {met_final['n_iter']} | avisos de convergência: "
      f"{met_final['avisos_convergencia']}")

print("\nClassification report:")
print(classification_report(y_teste, y_pred, target_names=NOMES_CLASSES, digits=4))

print("Matriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(confusion_matrix(y_teste, y_pred),
                   index=[f"real {c}" for c in NOMES_CLASSES],
                   columns=[f"prev {c}" for c in NOMES_CLASSES]).to_string())
print(f"\nMalignos classificados como benignos (falsos negativos): {met_final['FN']} "
      f"de {met_final['FN'] + met_final['VP']}")
print(f"Benignos classificados como malignos (falsos positivos):  {met_final['FP']} "
      f"de {met_final['FP'] + met_final['VN']}")

# Coerência com a validação cruzada
diferenca = met_final["acuracia"] - ACC_CV_REFERENCIA
print(f"\nAcurácia no teste {met_final['acuracia']:.4f} x validação cruzada "
      f"{ACC_CV_REFERENCIA:.4f} ± {DESVIO_CV_REFERENCIA:.4f} -> diferença {diferenca:+.4f} "
      f"({diferenca / DESVIO_CV_REFERENCIA:+.1f} desvios da CV)")

# ------------------------------------------------------------------
# 3. Figuras: matriz de confusão e curva ROC
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")

fig, ax = plt.subplots(figsize=(5.5, 4.8))
ConfusionMatrixDisplay.from_predictions(
    y_teste, y_pred, display_labels=NOMES_CLASSES, cmap="Blues", colorbar=False, ax=ax,
    text_kw={"fontsize": 16},
)
ax.grid(False)
ax.set_title("Matriz de confusão - MLP final (teste, n=171)")
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe real")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "matriz_confusao.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
# roc_curve + plot direto (a API de estilo do RocCurveDisplay muda entre versões)
fpr, tpr, _ = roc_curve(y_teste, modelo.predict_proba(X_teste)[:, 1])
ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
        label=f"MLP final (AUC = {met_final['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Aleatório (AUC = 0,5)")
ax.set_title("Curva ROC - MLP final (teste)")
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "curva_roc.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 4. Robustez à inicialização: sementes 0 a 9 (mesmo split, só muda o MLP)
# ------------------------------------------------------------------
linhas_sementes = []
for s in range(10):
    _, met = treinar_e_avaliar(mlp_final(s), X_treino, y_treino, X_teste, y_teste)
    linhas_sementes.append({"semente": s, **met})
sementes = pd.DataFrame(linhas_sementes).set_index("semente")

secao("ROBUSTEZ À INICIALIZAÇÃO (random_state 0 a 9, mesmo teste)")
colunas_metricas = ["acuracia", "precisao", "recall", "f1", "roc_auc"]
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 150):
    print(sementes[colunas_metricas + ["FN", "FP", "n_iter", "avisos_convergencia"]].to_string())
    print("\nMédia ± desvio:")
    for c in colunas_metricas:
        print(f"  {c:<10} {sementes[c].mean():.4f} ± {sementes[c].std():.4f}  "
              f"(mín {sementes[c].min():.4f}, máx {sementes[c].max():.4f})")
    print(f"  FN (malignos perdidos): de {sementes['FN'].min()} a {sementes['FN'].max()}")

# Monta o CSV do modelo final: semente 42, cada semente 0-9, média e desvio
tabela_final = pd.concat([
    pd.DataFrame([met_final], index=["semente_42 (modelo final)"]),
    sementes.set_axis([f"semente_{s}" for s in sementes.index]),
    sementes.mean().to_frame("media_sementes_0a9").T,
    sementes.std().to_frame("desvio_sementes_0a9").T,
])
tabela_final.index.name = "execucao"
tabela_final.to_csv(DIR_RESULTADOS / "modelo_final.csv", float_format="%.6f")

# ------------------------------------------------------------------
# 5. Comparação de referência no mesmo teste
# ------------------------------------------------------------------
# O MLP sem normalização usa max_iter=2000 (igual ao final) para que a única
# diferença entre os dois seja o StandardScaler.
referencias = {
    "MLP final (StandardScaler)": mlp_final(),
    "Regressão Logística (StandardScaler)": Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000)),
    ]),
    "MLP sem normalização": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000,
                                          random_state=SEMENTE),
}
linhas_comp = []
for nome, mod in referencias.items():
    _, met = treinar_e_avaliar(mod, X_treino, y_treino, X_teste, y_teste)
    linhas_comp.append({"modelo": nome, **met})
comparacao = pd.DataFrame(linhas_comp).set_index("modelo")
comparacao.to_csv(DIR_RESULTADOS / "comparacao_teste.csv", float_format="%.6f")

secao("COMPARAÇÃO NO TESTE")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
    print(comparacao[colunas_metricas + ["FN", "FP", "avisos_convergencia"]].to_string())

# Gráfico de barras agrupadas: acurácia, recall e F1 de cada modelo
longo = (comparacao[["acuracia", "recall", "f1"]]
         .rename(columns={"acuracia": "Acurácia", "recall": "Recall", "f1": "F1"})
         .reset_index().melt(id_vars="modelo", var_name="métrica", value_name="valor"))
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=longo, x="métrica", y="valor", hue="modelo",
            palette=["#4C72B0", "#55A868", "#C44E52"], ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
ax.set_ylim(max(0, longo["valor"].min() - 0.05), 1.01)  # zoom para ver as diferenças
ax.set_title("Desempenho no conjunto de teste (n=171)")
ax.set_xlabel("")
ax.set_ylabel("Valor (eixo y não começa em 0)")
# Legenda abaixo do gráfico para não cobrir as barras
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9,
          frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "comparacao_teste.png", dpi=150)
plt.close(fig)

secao("ARQUIVOS SALVOS")
for caminho in (DIR_RESULTADOS / "modelo_final.csv", DIR_RESULTADOS / "comparacao_teste.csv",
                DIR_FIGURAS / "matriz_confusao.png", DIR_FIGURAS / "curva_roc.png",
                DIR_FIGURAS / "comparacao_teste.png"):
    print(f"  {caminho}")
