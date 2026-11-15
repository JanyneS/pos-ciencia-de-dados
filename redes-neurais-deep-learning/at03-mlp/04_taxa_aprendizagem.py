"""
04_taxa_aprendizagem.py
Avalia o efeito da taxa de aprendizagem (learning_rate_init) no MLP, comparando
os otimizadores SGD puro (sem momentum, taxa constante) e Adam.

Usa Pipeline StandardScaler + MLPClassifier((100,)) e validação cruzada
estratificada no conjunto de treino. O teste é separado apenas para manter
a mesma divisão 70/30, mas NÃO é usado neste script.

Saídas:
  - resultados/taxa_aprendizagem.csv          (métricas, n_iter_, perda final, avisos, status)
  - resultados/taxa_aprendizagem_avisos.txt   (avisos registrados por configuração)
  - figuras/taxa_acuracia.png                 (acurácia de validação x taxa)
  - figuras/taxa_curvas_perda_sgd.png         (loss_curve_ de cada taxa, SGD)
  - figuras/taxa_curvas_perda_adam.png        (loss_curve_ de cada taxa, Adam)
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem janela: apenas salva as figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
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
TAXAS = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
LIMIAR_ACC_BAIXA = 0.70  # abaixo disso consideramos que o modelo "não aprendeu"

# Parâmetros específicos de cada otimizador
OTIMIZADORES = {
    "sgd": {"solver": "sgd", "momentum": 0.0, "learning_rate": "constant"},
    "adam": {"solver": "adam"},
}

# ------------------------------------------------------------------
# 1. Dados e divisão treino/teste (mesma dos scripts anteriores)
# ------------------------------------------------------------------
X, y = carregar_dados(binario=True)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMENTE
)
# A partir daqui só X_treino / y_treino são usados.
print(f"Treino: {X_treino.shape[0]} amostras | Teste (reservado): {X_teste.shape[0]} amostras")


def criar_modelo(otimizador, taxa):
    """Pipeline StandardScaler + MLP com o otimizador e a taxa indicados."""
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=SEMENTE,
                        learning_rate_init=taxa, **OTIMIZADORES[otimizador])
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def separar_avisos(capturados):
    """Separa os avisos capturados em convergência, falha de ajuste e numéricos."""
    conv = [str(w.message) for w in capturados if issubclass(w.category, ConvergenceWarning)]
    falha = [str(w.message) for w in capturados if issubclass(w.category, FitFailedWarning)]
    # RuntimeWarning aparece quando há overflow / divisão por zero nos cálculos
    numer = [str(w.message) for w in capturados if issubclass(w.category, RuntimeWarning)]
    return conv, falha, numer


# ------------------------------------------------------------------
# 2. Validação cruzada + treino completo para cada configuração
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

linhas = []
curvas = {otim: {} for otim in OTIMIZADORES}  # loss_curve_ de cada (otimizador, taxa)
registro_avisos = []                           # linhas de texto para o arquivo de avisos

for otim in OTIMIZADORES:
    for taxa in TAXAS:
        nome = f"{otim} | lr={taxa:g}"

        # 2.1 Validação cruzada. error_score=np.nan faz com que um fold que quebre
        # (ex.: pesos virando NaN/inf) gere NaN em vez de interromper o script.
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            scores = cross_validate(criar_modelo(otim, taxa), X_treino, y_treino, cv=cv,
                                    scoring=METRICAS, return_train_score=True,
                                    error_score=np.nan)
        conv_cv, falha_cv, numer_cv = separar_avisos(capturados)
        folds_falhos = int(np.isnan(scores["test_accuracy"]).sum())

        linha = {"otimizador": otim, "taxa": taxa}
        for m in METRICAS:
            for conjunto, prefixo in (("treino", "train"), ("validacao", "test")):
                valores = scores[f"{prefixo}_{m}"]
                # nanmean/nanstd ignoram folds que falharam
                linha[f"{m}_{conjunto}_media"] = np.nanmean(valores) if folds_falhos < 10 else np.nan
                linha[f"{m}_{conjunto}_desvio"] = np.nanstd(valores) if folds_falhos < 10 else np.nan
        linha["fit_time_medio_s"] = scores["fit_time"].mean()
        linha["avisos_convergencia_cv"] = len(conv_cv)
        linha["folds_falhos_cv"] = folds_falhos

        # 2.2 Treino no conjunto de treino inteiro: n_iter_, perda final e loss_curve_
        n_iter, perda_final, erro_final = np.nan, np.nan, ""
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            try:
                modelo = criar_modelo(otim, taxa).fit(X_treino, y_treino)
                mlp = modelo.named_steps["mlp"]
                n_iter = mlp.n_iter_
                curva = np.asarray(mlp.loss_curve_, dtype=float)
                perda_final = curva[-1]
                curvas[otim][taxa] = curva
            except Exception as erro:  # divergência pode gerar ValueError no sklearn
                erro_final = f"{type(erro).__name__}: {erro}"
        conv_final, _, numer_final = separar_avisos(capturados)

        linha["n_iter_treino"] = n_iter
        linha["perda_final_treino"] = perda_final
        linha["aviso_convergencia_treino"] = len(conv_final)

        # 2.3 Diagnóstico da configuração
        problemas = []
        if folds_falhos:
            problemas.append(f"{folds_falhos} fold(s) falharam")
        if erro_final:
            problemas.append("treino completo falhou")
        elif not np.isfinite(perda_final):
            problemas.append("perda não finita")
        if np.isfinite(linha["accuracy_validacao_media"]) and \
                linha["accuracy_validacao_media"] < LIMIAR_ACC_BAIXA:
            problemas.append("acurácia muito baixa")
        if numer_cv or numer_final:
            problemas.append("overflow numérico")
        if problemas:
            status = "DIVERGIU: " + "; ".join(problemas)
        elif conv_cv or conv_final:
            status = "não convergiu em max_iter"
        else:
            status = "ok"
        linha["status"] = status
        linhas.append(linha)

        # Guarda os detalhes para o arquivo de avisos
        registro_avisos.append(f"{nome}: status = {status}")
        registro_avisos.append(f"  ConvergenceWarning: {len(conv_cv)} na CV, "
                               f"{len(conv_final)} no treino completo")
        for msg in sorted(set(falha_cv + numer_cv + numer_final)):
            registro_avisos.append(f"  - {msg.strip()[:300]}")
        if erro_final:
            registro_avisos.append(f"  - treino completo: {erro_final[:300]}")

        print(f"  {nome:<16} acc val = {linha['accuracy_validacao_media']:.4f}  "
              f"n_iter = {n_iter}  perda = {perda_final:.4g}  [{status}]")

tabela = pd.DataFrame(linhas)
tabela.to_csv(DIR_RESULTADOS / "taxa_aprendizagem.csv", index=False, float_format="%.6g")
with open(DIR_RESULTADOS / "taxa_aprendizagem_avisos.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(registro_avisos) + "\n")

# ------------------------------------------------------------------
# 3. Tabela resumida no terminal
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("RESULTADOS (validação cruzada 10 folds + treino completo, conjunto de treino)")
print("=" * 100)


def media_desvio(m, conjunto):
    """Formata 'média ± desvio' de uma métrica."""
    return (tabela[f"{m}_{conjunto}_media"].map("{:.4f}".format) + " ± "
            + tabela[f"{m}_{conjunto}_desvio"].map("{:.4f}".format))


resumo = pd.DataFrame({
    "otim": tabela["otimizador"],
    "taxa": tabela["taxa"].map("{:g}".format),
    "acc_treino": media_desvio("accuracy", "treino"),
    "acc_validacao": media_desvio("accuracy", "validacao"),
    "f1_validacao": media_desvio("f1", "validacao"),
    "recall_validacao": media_desvio("recall", "validacao"),
    "n_iter": tabela["n_iter_treino"],
    "perda_final": tabela["perda_final_treino"].map("{:.4g}".format),
    "avisos_cv": tabela["avisos_convergencia_cv"],
    "status": tabela["status"],
})
with pd.option_context("display.width", 250, "display.max_columns", None,
                       "display.max_colwidth", 60):
    print(resumo.to_string(index=False))
print(f"\nSalvo em: {DIR_RESULTADOS / 'taxa_aprendizagem.csv'}")
print(f"Avisos em: {DIR_RESULTADOS / 'taxa_aprendizagem_avisos.txt'}")

# Melhor combinação pela acurácia média de validação (ignora NaN).
# Lista todas as empatadas, já que idxmax mostraria só a primeira.
acc_max = tabela["accuracy_validacao_media"].max()
empatadas = tabela[np.isclose(tabela["accuracy_validacao_media"], acc_max)]
print("\nMelhor acurácia de validação:")
for _, m in empatadas.iterrows():
    print(f"  {m['otimizador']} com lr={m['taxa']:g} -> "
          f"{m['accuracy_validacao_media']:.4f} ± {m['accuracy_validacao_desvio']:.4f} "
          f"(recall {m['recall_validacao_media']:.4f}, {m['n_iter_treino']} épocas)")

# ------------------------------------------------------------------
# 4. Figuras
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
CORES_OTIM = {"sgd": "#C44E52", "adam": "#4C72B0"}
ROTULO_OTIM = {"sgd": "SGD (sem momentum)", "adam": "Adam"}

# 4.1 Acurácia de validação x taxa (escala log), uma linha por otimizador
fig, ax = plt.subplots(figsize=(8, 5))
for otim in OTIMIZADORES:
    sub = tabela[tabela["otimizador"] == otim]
    media = sub["accuracy_validacao_media"]
    desvio = sub["accuracy_validacao_desvio"]
    ax.plot(sub["taxa"], media, marker="o", linewidth=1.8,
            color=CORES_OTIM[otim], label=ROTULO_OTIM[otim])
    ax.fill_between(sub["taxa"], media - desvio, media + desvio,
                    color=CORES_OTIM[otim], alpha=0.2)
ax.set_xscale("log")
ax.set_xticks(TAXAS)
ax.set_xticklabels([f"{t:g}" for t in TAXAS])
ax.minorticks_off()
ax.set_title("Acurácia de validação x taxa de aprendizagem (± desvio, 10 folds)")
ax.set_xlabel("learning_rate_init (escala log)")
ax.set_ylabel("Acurácia de validação")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "taxa_acuracia.png", dpi=150)
plt.close(fig)

# 4.2 Curvas de perda sobrepostas, uma figura por otimizador.
# Paleta sequencial: cores mais escuras = taxas maiores.
cores_taxa = dict(zip(TAXAS, sns.color_palette("viridis_r", len(TAXAS))))
for otim in OTIMIZADORES:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for taxa in TAXAS:
        if taxa not in curvas[otim]:
            continue  # treino completo falhou: não há curva para desenhar
        curva = curvas[otim][taxa].copy()
        curva[~np.isfinite(curva)] = np.nan  # NaN/inf não são desenhados
        ax.plot(np.arange(1, len(curva) + 1), curva, color=cores_taxa[taxa],
                linewidth=1.5, label=f"lr={taxa:g} ({len(curva)} épocas)")
    ax.set_xscale("log")  # épocas variam de dezenas a milhares
    ax.set_yscale("log")
    ax.set_title(f"Curvas de perda no treino - {ROTULO_OTIM[otim]}")
    ax.set_xlabel("Época (escala log)")
    ax.set_ylabel("Perda log-loss (escala log)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"taxa_curvas_perda_{otim}.png", dpi=150)
    plt.close(fig)

print("\nFiguras salvas:")
for arq in ("taxa_acuracia.png", "taxa_curvas_perda_sgd.png", "taxa_curvas_perda_adam.png"):
    print(f"  {DIR_FIGURAS / arq}")
