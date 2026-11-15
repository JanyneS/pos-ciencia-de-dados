"""
03_neuronios.py
Avalia o efeito do número de neurônios (e de uma segunda camada oculta) no MLP,
usando Pipeline StandardScaler + MLPClassifier e validação cruzada estratificada
no conjunto de treino.

O conjunto de teste é separado apenas para manter a mesma divisão 70/30,
mas NÃO é usado neste script.

Saídas:
  - resultados/neuronios.csv            (treino e validação: média e desvio; fit_time)
  - resultados/neuronios_avisos.txt     (ConvergenceWarnings registrados)
  - figuras/neuronios_acuracia.png      (acurácia x nº de neurônios, uma camada)
  - figuras/neuronios_recall.png        (recall x nº de neurônios, uma camada)
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem janela: apenas salva as figuras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Arquiteturas testadas: tuplas no formato de hidden_layer_sizes
UMA_CAMADA = [(n,) for n in (1, 2, 3, 5, 10, 20, 50, 100, 200)]
DUAS_CAMADAS = [(10, 10), (20, 10), (50, 25), (100, 50)]
ARQUITETURAS = UMA_CAMADA + DUAS_CAMADAS

# ------------------------------------------------------------------
# 1. Dados e divisão treino/teste (mesma do 02_normalizacao.py)
# ------------------------------------------------------------------
X, y = carregar_dados(binario=True)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMENTE
)
# A partir daqui só X_treino / y_treino são usados.
print(f"Treino: {X_treino.shape[0]} amostras | Teste (reservado): {X_teste.shape[0]} amostras")

# ------------------------------------------------------------------
# 2. Validação cruzada para cada arquitetura
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

linhas = []
avisos = {}  # mensagens de ConvergenceWarning por arquitetura

for camadas in ARQUITETURAS:
    nome = str(camadas).replace(" ", "")  # ex.: "(10,)" ou "(50,25)"
    modelo = Pipeline([
        ("scaler", StandardScaler()),  # ajustado só no fold de treino (sem vazamento)
        ("mlp", MLPClassifier(hidden_layer_sizes=camadas, max_iter=2000,
                              random_state=SEMENTE)),
    ])

    # Captura os avisos de convergência em vez de só imprimi-los
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        scores = cross_validate(modelo, X_treino, y_treino, cv=cv,
                                scoring=METRICAS, return_train_score=True)
    avisos[nome] = [str(w.message) for w in capturados
                    if issubclass(w.category, ConvergenceWarning)]

    linha = {
        "arquitetura": nome,
        "n_camadas": len(camadas),
        "neuronios_total": sum(camadas),
    }
    for m in METRICAS:
        for conjunto, prefixo in (("treino", "train"), ("validacao", "test")):
            valores = scores[f"{prefixo}_{m}"]
            linha[f"{m}_{conjunto}_media"] = valores.mean()
            linha[f"{m}_{conjunto}_desvio"] = valores.std()
    linha["fit_time_medio_s"] = scores["fit_time"].mean()
    linha["avisos_convergencia_cv"] = len(avisos[nome])
    linhas.append(linha)
    print(f"  {nome:<10} acc validação = {linha['accuracy_validacao_media']:.4f}  "
          f"({linha['fit_time_medio_s']:.2f} s/fold)")

tabela = pd.DataFrame(linhas).set_index("arquitetura")
tabela.to_csv(DIR_RESULTADOS / "neuronios.csv", float_format="%.6f")

# ------------------------------------------------------------------
# 3. Tabela resumida no terminal
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTADOS DA VALIDAÇÃO CRUZADA (10 folds, conjunto de treino)")
print("=" * 80)
resumo = pd.DataFrame(index=tabela.index)
for m in METRICAS:
    for conjunto in ("treino", "validacao"):
        resumo[f"{m}_{conjunto}"] = (
            tabela[f"{m}_{conjunto}_media"].map("{:.4f}".format) + " ± "
            + tabela[f"{m}_{conjunto}_desvio"].map("{:.4f}".format)
        )
# Diferença treino - validação na acurácia: indicador de overfitting
resumo["gap_acc"] = (tabela["accuracy_treino_media"]
                     - tabela["accuracy_validacao_media"]).map("{:+.4f}".format)
resumo["fit_s"] = tabela["fit_time_medio_s"].map("{:.2f}".format)
resumo["avisos"] = tabela["avisos_convergencia_cv"]
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(resumo.to_string())
print(f"\nSalvo em: {DIR_RESULTADOS / 'neuronios.csv'}")

# Melhor configuração pela acurácia média de validação
# (em caso de empate, idxmax pega a primeira, ou seja, a mais simples da lista)
melhor = tabela["accuracy_validacao_media"].idxmax()
print(f"\nMelhor acurácia de validação: {melhor} = "
      f"{tabela.loc[melhor, 'accuracy_validacao_media']:.4f} ± "
      f"{tabela.loc[melhor, 'accuracy_validacao_desvio']:.4f}")

# ------------------------------------------------------------------
# 4. Registro dos ConvergenceWarnings
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("AVISOS DE CONVERGÊNCIA (ConvergenceWarning)")
print("=" * 80)
total_avisos = sum(len(msgs) for msgs in avisos.values())
with open(DIR_RESULTADOS / "neuronios_avisos.txt", "w", encoding="utf-8") as f:
    for nome, msgs in avisos.items():
        f.write(f"{nome}: {len(msgs)} aviso(s)\n")
        for msg in msgs:
            f.write(f"  - {msg}\n")
        if msgs:
            print(f"{nome}: {len(msgs)} aviso(s) | exemplo: {msgs[0]}")
print(f"Total: {total_avisos} aviso(s)")
print(f"Salvo em: {DIR_RESULTADOS / 'neuronios_avisos.txt'}")

# ------------------------------------------------------------------
# 5. Figuras (apenas arquiteturas de uma camada)
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
CORES = {"treino": "#4C72B0", "validacao": "#DD8452"}
ROTULOS = {"treino": "Treino", "validacao": "Validação"}
uma = tabela[tabela["n_camadas"] == 1].sort_values("neuronios_total")


def grafico_metrica(metrica, titulo, arquivo):
    """Linhas de treino e validação com faixa de ± desvio padrão, eixo x em log."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = uma["neuronios_total"]
    for conjunto in ("treino", "validacao"):
        media = uma[f"{metrica}_{conjunto}_media"]
        desvio = uma[f"{metrica}_{conjunto}_desvio"]
        ax.plot(x, media, marker="o", color=CORES[conjunto], linewidth=1.8,
                label=ROTULOS[conjunto])
        ax.fill_between(x, media - desvio, media + desvio,
                        color=CORES[conjunto], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(x)  # mostra 1, 2, 3, 5, ... em vez de 10^0, 10^1
    ax.minorticks_off()
    ax.set_title(titulo)
    ax.set_xlabel("Nº de neurônios na camada oculta (escala log)")
    ax.set_ylabel(metrica.capitalize() if metrica != "accuracy" else "Acurácia")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / arquivo, dpi=150)
    plt.close(fig)
    print(f"  {DIR_FIGURAS / arquivo}")


print("\nFiguras salvas:")
grafico_metrica("accuracy", "Acurácia x nº de neurônios (1 camada oculta, ± desvio)",
                "neuronios_acuracia.png")
grafico_metrica("recall", "Recall (Maligno) x nº de neurônios (1 camada oculta, ± desvio)",
                "neuronios_recall.png")
