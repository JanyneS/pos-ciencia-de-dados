# ============================================================
# PERGUNTA 4: A renda influencia na evasão?
# Variáveis: Renda Familiar (quantitativa contínua) + Status (qualitativa nominal)
# Medidas usadas:
#   - Posição: média e mediana da renda separadas por grupo
#   - Dispersão: desvio padrão e variância da renda por grupo
# 
# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv("dados.csv")

renda_evadidos = df[df["Status"] == "Evadido"]["Renda Familiar"]
renda_ativos   = df[df["Status"] == "Ativo"]["Renda Familiar"]
renda_formados = df[df["Status"] == "Formado"]["Renda Familiar"]

# --- 3. Medidas de Posição por grupo ---

# MÉDIA:
media_evadidos = renda_evadidos.mean()
media_ativos   = renda_ativos.mean()
media_formados = renda_formados.mean()

# MEDIANA: valor central da renda quando ordenada
mediana_evadidos = renda_evadidos.median()
mediana_ativos   = renda_ativos.median()
mediana_formados = renda_formados.median()

# --- 4. Medidas de Dispersão por grupo ---

# DESVIO PADRÃO:
dp_evadidos = renda_evadidos.std()
dp_ativos   = renda_ativos.std()

# VARIÂNCIA: 
var_evadidos = renda_evadidos.var()
var_ativos   = renda_ativos.var()


print("=" * 55)
print("PERGUNTA 4: A renda familiar influencia na evasão?")
print("=" * 55)
print()
print(f"{'Grupo':<12} {'Média (R$)':>12} {'Mediana (R$)':>13} {'Desv. Padrão':>13} {'Variância':>12}")
print("-" * 65)
print(f"{'Evadidos':<12} {media_evadidos:>12.2f} {mediana_evadidos:>13.2f} {dp_evadidos:>13.2f} {var_evadidos:>12.2f}")
print(f"{'Ativos':<12} {media_ativos:>12.2f} {mediana_ativos:>13.2f} {dp_ativos:>13.2f} {var_ativos:>12.2f}")
print(f"{'Formados':<12} {media_formados:>12.2f} {media_formados:>13.2f} {'—':>13} {'—':>12}")
print()


diferenca_media = media_ativos - media_evadidos
print(f"Diferença de média (Ativo - Evadido): R$ {diferenca_media:.2f}")
print()


print("--- Interpretação ---")
print(f"Evadidos têm renda média de R$ {media_evadidos:.2f}.")
print(f"Ativos têm renda média de R$ {media_ativos:.2f}.")
if diferenca_media > 200:
    print("A diferença sugere que renda menor está associada à evasão.")
elif diferenca_media < -200:
    print("Surpreendentemente, evadidos têm renda maior — outros fatores podem explicar.")
else:
    print("A diferença é pequena — renda sozinha pode não ser o principal fator.")
print()
print(f"O desvio padrão dos evadidos (R$ {dp_evadidos:.2f}) mostra que")
print(f"as rendas são bastante variadas dentro desse grupo.")
