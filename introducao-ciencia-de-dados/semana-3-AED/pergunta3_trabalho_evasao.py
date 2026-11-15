# ============================================================
# PERGUNTA 3: Estudantes que trabalham têm maior probabilidade de evadir?
# Variáveis: Trabalha Atualmente (qualitativa nominal) + Status (qualitativa nominal)
# Medidas usadas:
#   - Posição: moda (variáveis qualitativas não usam média/mediana)
#   - Dispersão: covariância (relação entre duas variáveis)

# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv("dados.csv")

evadidos_trabalham = df[(df["Status"] == "Evadido") & (df["Trabalha Atualmente"] == "Sim")]
evadidos_nao_trabalham = df[(df["Status"] == "Evadido") & (df["Trabalha Atualmente"] == "Nao")]

total_trabalham = df[df["Trabalha Atualmente"] == "Sim"]
total_nao_trabalham = df[df["Trabalha Atualmente"] == "Nao"]

# --- 3. Moda ---
# Qual o status mais frequente entre quem trabalha?
moda_trabalham = df[df["Trabalha Atualmente"] == "Sim"]["Status"].mode()[0]

# Qual o status mais frequente entre quem não trabalha?
moda_nao_trabalham = df[df["Trabalha Atualmente"] == "Nao"]["Status"].mode()[0]

# --- 4. Covariância ---
df["trabalha_num"] = df["Trabalha Atualmente"].map({"Sim": 1, "Nao": 0})
df["evadido_num"] = df["Status"].apply(lambda x: 1 if x == "Evadido" else 0)

covariancia = df["trabalha_num"].cov(df["evadido_num"])

# --- 5. Taxa de evasão por grupo (porcentagem) ---
taxa_evasao_trabalham = (len(evadidos_trabalham) / len(total_trabalham)) * 100
taxa_evasao_nao_trabalham = (len(evadidos_nao_trabalham) / len(total_nao_trabalham)) * 100

print("=" * 55)
print("PERGUNTA 3: Estudantes que trabalham evadem mais?")
print("=" * 55)
print()
print("--- Contagem ---")
print(f"Evadidos que trabalham:     {len(evadidos_trabalham)}")
print(f"Evadidos que não trabalham: {len(evadidos_nao_trabalham)}")
print()
print("--- Medidas de Posição (Moda) ---")
print(f"Status mais comum entre quem trabalha:     {moda_trabalham}")
print(f"Status mais comum entre quem não trabalha: {moda_nao_trabalham}")
print()
print("--- Medida de Dispersão (Covariância) ---")
print(f"Covariância entre trabalho e evasão: {covariancia:.4f}")
print()
print("--- Taxa de Evasão por Grupo ---")
print(f"Taxa de evasão entre quem trabalha:     {taxa_evasao_trabalham:.1f}%")
print(f"Taxa de evasão entre quem não trabalha: {taxa_evasao_nao_trabalham:.1f}%")
print()

print("--- Interpretação ---")
if covariancia > 0:
    print("Covariância POSITIVA: quem trabalha tende a evadir mais.")
elif covariancia < 0:
    print("Covariância NEGATIVA: quem trabalha tende a evadir menos.")
else:
    print("Covariância ZERO: não há relação entre trabalhar e evadir.")
