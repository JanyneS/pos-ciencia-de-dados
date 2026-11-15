"""
02_normalizacao.py
Compara o MLP treinado com os dados originais e com dados padronizados
(StandardScaler), usando validação cruzada estratificada no conjunto de treino.

O conjunto de teste é separado aqui apenas para fixar a divisão 70/30,
mas NÃO é usado neste script.

Saídas:
  - resultados/normalizacao.csv              (média e desvio de cada métrica)
  - resultados/normalizacao_avisos.txt       (ConvergenceWarnings registrados)
  - figuras/normalizacao_boxplot.png         (acurácia por fold)
  - figuras/normalizacao_curva_perda.png     (loss_curve_ no treino)
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem janela: apenas salva as figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import DIR_FIGURAS, DIR_RESULTADOS, carregar_dados, criar_pastas

# Garante acentuação correta no terminal do Windows
sys.stdout.reconfigure(encoding="utf-8")
criar_pastas()

SEMENTE = 42
METRICAS = ["accuracy", "f1", "recall"]  # f1 e recall da classe positiva (1 = Maligno)

# ------------------------------------------------------------------
# 1. Dados e divisão treino/teste
# ------------------------------------------------------------------
X, y = carregar_dados(binario=True)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMENTE
)
# A partir daqui só X_treino / y_treino são usados.
print(f"Treino: {X_treino.shape[0]} amostras | Teste (reservado): {X_teste.shape[0]} amostras")
print(f"Proporção de malignos no treino: {y_treino.mean():.3f}")

# ------------------------------------------------------------------
# 2. Modelos
# ------------------------------------------------------------------
def novo_mlp():
    """MLP com a mesma configuração nos dois cenários."""
    return MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=SEMENTE)


modelos = {
    "Sem normalização": novo_mlp(),
    "Com StandardScaler": Pipeline([
        ("scaler", StandardScaler()),  # ajustado só no fold de treino (sem vazamento)
        ("mlp", novo_mlp()),
    ]),
}

# ------------------------------------------------------------------
# 3. Validação cruzada (10 folds estratificados) com registro de avisos
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

resultados_cv = {}   # scores por fold de cada modelo
avisos = {}          # mensagens de ConvergenceWarning por modelo

for nome, modelo in modelos.items():
    # catch_warnings(record=True) guarda os avisos em vez de só imprimi-los.
    # n_jobs=None (padrão) roda os folds no mesmo processo, então os avisos são capturados.
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        scores = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=METRICAS)
    resultados_cv[nome] = scores
    avisos[nome] = [str(w.message) for w in capturados
                    if issubclass(w.category, ConvergenceWarning)]

# ------------------------------------------------------------------
# 4. Tabela de resultados: média e desvio padrão de cada métrica
# ------------------------------------------------------------------
linhas = []
for nome, scores in resultados_cv.items():
    linha = {"modelo": nome}
    for m in METRICAS:
        valores = scores[f"test_{m}"]
        linha[f"{m}_media"] = valores.mean()
        linha[f"{m}_desvio"] = valores.std()
    # Nº de folds (de 10) em que o MLP não convergiu em max_iter iterações
    linha["avisos_convergencia_cv"] = len(avisos[nome])
    linhas.append(linha)

tabela = pd.DataFrame(linhas).set_index("modelo")
tabela.to_csv(DIR_RESULTADOS / "normalizacao.csv", float_format="%.6f")

print("\n" + "=" * 60)
print("RESULTADOS DA VALIDAÇÃO CRUZADA (10 folds, conjunto de treino)")
print("=" * 60)
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(tabela.to_string())
print(f"\nSalvo em: {DIR_RESULTADOS / 'normalizacao.csv'}")

# ------------------------------------------------------------------
# 5. Treino final de cada modelo em todo o conjunto de treino (para a loss_curve_)
# ------------------------------------------------------------------
curvas = {}
for nome, modelo in modelos.items():
    modelo_final = clone(modelo)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo_final.fit(X_treino, y_treino)
    avisos[nome] += [f"[treino completo] {w.message}" for w in capturados
                     if issubclass(w.category, ConvergenceWarning)]

    # No Pipeline, o MLP fica no passo "mlp"
    mlp = modelo_final.named_steps["mlp"] if isinstance(modelo_final, Pipeline) else modelo_final
    curvas[nome] = mlp.loss_curve_
    print(f"{nome}: {mlp.n_iter_} iterações, perda final = {mlp.loss_curve_[-1]:.4f}")

# ------------------------------------------------------------------
# 6. Registro dos ConvergenceWarnings
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("AVISOS DE CONVERGÊNCIA (ConvergenceWarning)")
print("=" * 60)
with open(DIR_RESULTADOS / "normalizacao_avisos.txt", "w", encoding="utf-8") as f:
    for nome, msgs in avisos.items():
        resumo = f"{nome}: {len(msgs)} aviso(s)"
        print(resumo)
        f.write(resumo + "\n")
        for msg in msgs:
            f.write(f"  - {msg}\n")
        if msgs:
            print(f"  exemplo: {msgs[0]}")
print(f"\nSalvo em: {DIR_RESULTADOS / 'normalizacao_avisos.txt'}")

# ------------------------------------------------------------------
# 7. Figuras
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
CORES = {"Sem normalização": "#C44E52", "Com StandardScaler": "#4C72B0"}

# 7.1 Boxplot da acurácia por fold (com os pontos de cada fold sobrepostos)
acc_folds = pd.DataFrame({
    "modelo": np.repeat(list(resultados_cv), 10),
    "acuracia": np.concatenate([s["test_accuracy"] for s in resultados_cv.values()]),
})
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=acc_folds, x="modelo", y="acuracia", hue="modelo", palette=CORES,
            width=0.5, linewidth=1, fliersize=0, legend=False, ax=ax)
sns.stripplot(data=acc_folds, x="modelo", y="acuracia", color="black",
              size=5, alpha=0.6, jitter=0.08, ax=ax)
ax.set_title("Acurácia por fold (StratifiedKFold, k=10)")
ax.set_xlabel("")
ax.set_ylabel("Acurácia")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "normalizacao_boxplot.png", dpi=150)
plt.close(fig)

# 7.2 Curva de perda no treino. Escala log no eixo y, pois sem normalização
# a perda inicial pode ser ordens de grandeza maior.
fig, ax = plt.subplots(figsize=(8, 5))
for nome, curva in curvas.items():
    ax.plot(np.arange(1, len(curva) + 1), curva, label=f"{nome} ({len(curva)} iterações)",
            color=CORES[nome], linewidth=1.8)
ax.set_yscale("log")
ax.set_title("Curva de perda (loss_curve_) no conjunto de treino")
ax.set_xlabel("Iteração (época)")
ax.set_ylabel("Perda log-loss (escala log)")
ax.legend()
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "normalizacao_curva_perda.png", dpi=150)
plt.close(fig)

print("\nFiguras salvas:")
print(f"  {DIR_FIGURAS / 'normalizacao_boxplot.png'}")
print(f"  {DIR_FIGURAS / 'normalizacao_curva_perda.png'}")
